# 算法文档 —— 基于 CUDA 的三维脏器掩膜后处理流水线

本文档逐步讲解完整的处理流程，以及流程中使用的**每一个算法**，与
[`../src`](../src) 中的代码一一对应。（英文版：[`ALGORITHM.md`](ALGORITHM.md)；
CUDA 知识详解：[`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md)）

> **任务（原始需求）。** *将逐层分割掩膜重建为三维体数据，实现基于 CUDA 的后处理流水线——
> 最大连通域标记与筛选去除层间孤立假阳性区域，3D 形态学开闭弥合层间不连续并消除孔洞毛刺，
> 3D 高斯平滑抑制边缘噪声，输出规整的脏器三维掩膜，支撑后续体积统计与三维可视化。*

概括来说：先把逐层（每张轴向 CT 切片）的 2D 分割结果重建为三维体数据，再依次执行
CUDA 后处理流水线——(1) 连通域标记与体积筛选，去除层间孤立假阳性；(2) 三维形态学
开/闭运算，弥合层间不连续并消除孔洞与毛刺；(3) 三维高斯平滑抑制边缘噪声——最终输出
规整的脏器三维掩膜，支撑体积统计与三维可视化。

---

## 0. 数据表示与符号约定

| 符号 | 含义 |
|---|---|
| `W, H, D` | 体数据的宽 (x)、高 (y)、层数/深度 (z) |
| `N = W·H·D` | 体素总数 |
| `dx, dy, dz` | 体素间距（毫米，来自 `meta.txt`） |
| `M[i] ∈ {0,1}` | 二值掩膜，`uint8` 存储，1 = 前景（脏器） |

**内存布局（行主序，z 变化最慢）：**

$\[ i = (z \cdot H + y)\cdot W + x \]$

所有三维核函数通过 `dim3 block(8,8,8)` / `gridFor(W,H,D)` 网格把线程映射到
`(x,y,z)`（见 [`common.cuh`](../src/common.cuh)）；所有一维核函数在 `N` 上使用 256
线程。边界访问采用**复制填充（clamp）**：`clampi(v, 0, dim-1)`。

**流水线顺序**（见 [`main.cu`](../src/main.cu)）：

```
切片 → [1] 三维重建 → [2] 连通域+筛选 → [3] 闭+开运算 → [4] 高斯平滑 → [5] 收尾输出
```

除阶段 1（生成掩膜）和阶段 5（读取掩膜）外，各阶段都在设备端掩膜 `d_mask` 上
**原地（in-place）** 处理。

---

## 阶段 1 —— 由逐层掩膜重建三维体数据

**文件：** [`volume.cu`](../src/volume.cu) · **入口：** `reconstructOnDevice()`

### 目的
把 2D 分割网络（例如对每张轴向 CT 切片运行的 UNet）逐层输出的结果，拼接并归一化为
一个三维二值体数据。

### 输入格式
- `data/slices/slice_%04d.u8` —— 每张切片一个 `W·H` 字节的原始文件（取值 `0/1` 或
  `0..255`）；也可用 `--stack` 指定单个连续堆栈文件。
- `data/meta.txt` —— `W H D dx dy dz`。

### 算法
1. **主机端拼接**（`loadSlicesDir`）：顺序读取 `D` 个切片文件，写入同一主机缓冲区
   `h_raw[z·W·H … ]`。这一步就是字面意义上的“逐层堆叠”。
2. **上传 + 二值化**（`reconstructOnDevice`）：
   - `cudaMalloc` 分配 `d_raw`、`d_mask`（各 N 字节）；`cudaMemcpy` 将 `h_raw → d_raw`。
   - `binarizeKernel`：对每个体素，`d_mask[i] = (d_raw[i] > binThreshold) ? 1 : 0`。
   - 释放 `d_raw`，返回 `d_mask`。

```
binarizeKernel(raw, out, N, thr):
    i = 全局线程id
    if i < N: out[i] = (raw[i] > thr)
```

### 开销
- 时间：`O(N)` 的访存受限拷贝 + `O(N)` 二值化；主要受主机文件 I/O 与 H2D PCIe 传输支配。
- 显存：临时占用 `2·N` 字节。

此处的阈值化还能把各种掩膜编码（例如从 PNG 导出的 `0/255` 切片）统一归一化为干净的
`{0,1}` 体数据，供后续阶段使用。

---

## 阶段 2 —— 三维连通域标记 + 体积筛选

**文件：** [`ccl.cu`](../src/ccl.cu) · **入口：** `filterComponents()`

### 目的
标记掩膜中每一个三维连通区域，统计各区域体积，并删除过小的区域——这些正是
**层间孤立假阳性**（只在一两张切片上出现、且与脏器主体不相连的斑点）。

### 为什么在 GPU 上用并查集
经典串行 CCL（两遍扫描 + 合并表）本质上是串行的。在 GPU 上我们采用**并行并查集
（union-find）**：每个体素自成集合；相邻前景体素做 union；再用路径压缩压平森林。
一个连通域的规范标签取其中的**最小体素索引**，这既给出自然稳定的 id，也保证了收敛
（标签只会不断变小）。

### 连通性
采用 **26 连通**（完整 3×3×3 邻域去掉中心）。为使每条无向边只处理一次，merge 核只访问
**13 个前向偏移**：

```
{+1,0,0} {+1,+1,0} {+1,-1,0} {0,+1,0} {0,-1,0} {-1,+1,0} {-1,-1,0}
{+1,0,+1} {+1,+1,+1} {+1,-1,+1} {0,+1,+1} {0,-1,+1} {0,0,+1}
```

### 数据结构
- `parent[N]`（`int`）—— 并查集森林，初始 `parent[i] = i`。
- `counts[N]`（`int`）—— 按根索引的体积直方图。
- `DevStats` —— 若干设备端标量，用原子操作归约得到。

### 核函数序列

**a. `initParentKernel`** —— 对所有 `i` 置 `parent[i] = i`（一维遍历 `N`）。

**b. `mergeKernel`**（三维遍历前景体素）—— 与前向邻居做 union：

```
i = 体素; if !M[i] return
ri = find(parent, i)
对每个前向偏移 o:
    j = neighbour(i, o); if 越界 or !M[j] continue
    rj = find(parent, j)
    if ri != rj:
        lo = min(ri, rj); hi = max(ri, rj)
        atomicMin(&parent[hi], lo)     # 把较大的根挂到较小的根
        ri = lo
        changed = 1
```

其中 `find`（`uf_find`）是**只读**的迭代查根——它不做压缩，因此 merge 阶段多线程之间
不会发生写竞争。`atomicMin` 使并发下的 union 安全且单调：根的值只会减小，从而保证迭代
必然收敛。

**c. `compressKernel`**（一维遍历 `N`）—— 指针跳跃（path jumping）：

```
r = find(parent, i)
if parent[i] != r: parent[i] = r; changed = 1
```

**d. 收敛循环**（主机端）：把 `changed` 复位为 0，跑 merge，再跑 compress，读回
`changed`；重复直到没有任何 parent 改变（安全上限 500 次）。压缩每轮会把路径长度减半，
因此实际迭代次数很少（**实测 2–4 次**，肝脏体数据为 3 次）。

**e. `buildCountsKernel`**（三维）—— 体积直方图：

```
if M[i]: atomicAdd(&counts[ find(parent, i) ], 1)
```

收敛后 `find(i)` 即该连通域的最小索引，因此同一连通域的所有体素都累加到同一个槽位。

**f. `statsKernel`**（一维遍历 `counts`）—— 一遍计算出：
- `numComp`   —— 连通域数量（`counts[i] > 0`）；
- `totalFg`   —— Σ counts = 筛选前的前景体素数；
- `keptBySize`、`keptFg` —— 满足 `counts[i] ≥ minSize` 的连通域数 / 体素数；
- `packedMax` —— 用一次原子操作同时得到**最大连通域及其标签**：

\[ \text{key} = (\text{count} \ll 32)\;|\;\text{label}, \qquad \texttt{atomicMax}(\text{packedMax}, \text{key}) \]

由于 count 占高 32 位，对 `unsigned long long` 做 `atomicMax` 会选出体积最大的连通域；
低 32 位随即给出它的标签。（注意：CUDA **没有** `atomicAdd(long long*)` 重载——64 位
原子累加必须用 `unsigned long long`。）

**g. `filterKernel`**（三维）—— 原地应用筛选决策：

```
if !M[i] return
lab = find(parent, i)
keep = keepLargestOnly ? (lab == largestLabel) : (counts[lab] >= minSize)
if !keep: M[i] = 0
```

### 输出
`CclResult` 给出 `numComponentsTotal`、`numComponentsKept`、`largestComponent`、
`foregroundBefore`、`foregroundAfter`、`iterations`。

### 正确性论证
两个体素拥有相同的根 **当且仅当** 存在一条 26 连通的前景路径把它们连起来：merge 步骤
恰好对前景邻接图的每条边做 union，`atomicMin` 把每个连通子图收敛到其最小索引。压缩只
改变路径长度、不改变根。当没有任何根再变化时循环停止，即到达 union 操作的不动点。

### 开销
- 时间：merge/compress 为 `O(K · N · 13)`（K 为迭代次数，很小），加上 `O(N)` 的直方图/
  统计。访存受限。
- 显存：`parent` + `counts` = `8·N` 字节（8.39M 体素约 67 MB）。

---

## 阶段 3 —— 三维形态学闭运算 + 开运算

**文件：** [`morphology.cu`](../src/morphology.cu) · **入口：** `morphOpenClose()`

### 目的
- **闭运算**（先膨胀后腐蚀）：弥合**层间不连续**（某张切片上掩膜短暂缺失），并**填充
  内部孔洞**。
- **开运算**（先腐蚀后膨胀）：去除表面的**毛刺/细小尖突**等小突起。

流水线**先闭后开**：先把缺口补上，再修剪松散碎片。

### 二值形态学即 min/max 滤波
在 `{0,1}` 体数据上，给定结构元（SE）`B`：

\[ \text{erode}(M)(p) = \min_{o \in B} M(p+o), \qquad \text{dilate}(M)(p) = \max_{o \in B} M(p+o) \]

### 可分离的盒状结构元
我们使用带各轴半径 `(rx, ry, rz)` 的**盒状（box）** 结构元。盒是三段一维线段的
Minkowski 和，而 min/max 滤波是**可分离**的，所以三维盒腐蚀等价于沿 X、再沿 Y、再沿 Z
的三次一维腐蚀。这就把 `O(rx·ry·rz)` 的邻域降为 `O(rx+ry+rz)`。

**`morphAxisKernel`** —— 沿某一轴的一维 min（腐蚀）或 max（膨胀）：

```
acc = dilate ? 0 : 1
for d in [-r, +r]:
    (nx,ny,nz) = 沿 `axis` 对采样坐标做 clamp   # 复制填充
    v = in[idx]
    acc = dilate ? max(acc, v) : min(acc, v)
out[体素] = acc
```

**`separable(src, A, B, dst)`** 把三遍路由为 `X: src→A`、`Y: A→B`、`Z: B→dst`
（要求 `dst ≠ B`；`dst` 可以等于 `src` 或 `A`）。`morphOpenClose` 用两块临时缓冲
`t1, t2` 来回倒换，从而无需额外分配：

```
闭: separable(mask, t1, t2, t1, dilate);  separable(t1, t2, t1, mask, erode)
开: separable(mask, t1, t2, t1, erode);   separable(t1, t2, t1, mask, dilate)
```

### 弥合层间间隙
由于结构元半径是**逐轴**的，增大 `rz` 就能沿切片方向（z）拉长结构元。当两块已填充的
薄层之间只隔一张空切片时，只要 `rz ≥ 1` 即可闭合：膨胀让相邻薄层跨过间隙相互重叠，
随后的腐蚀会保留这个被桥接的体素（它的两个 z 邻居都是前景）。要桥接两张切片的间隙，
用 `rz = 2`，以此类推。

### 一个重要细节（由 self-test 验证）
沿**单一轴**的一维开运算无法去除*恰好沿着该轴*的毛刺：腐蚀也许削掉了尖突的尖端，但
幸存的邻居又会被膨胀重新长回去（长度为 5 的一段相对尺寸 3 的结构元本就是“开”的）。
因此去除细尖突需要**三维盒状**结构元（`rx = ry = rz ≥ 1`）：腐蚀会整体删掉只有 1 体素
粗的尖突（其 3×3×3 邻域并非全为前景），膨胀也无法把它重建出来。这就是默认半径取
`1 1 1` 的原因。

### 开销
- 时间：每次腐蚀/膨胀为 `O(N · (rx+ry+rz))`；开+闭 = 4 次运算 × 3 遍 = **12 次核启动**。
- 显存：`2·N` 字节临时缓冲（`t1, t2`）。

---

## 阶段 4 —— 三维高斯平滑 + 阈值化

**文件：** [`gaussian.cu`](../src/gaussian.cu) · **入口：** `gaussianSmoothMask()`

### 目的
对掩膜做低通滤波以**抑制边缘噪声**（表面上的单体素抖动、斑点），得到平滑规整的边界；
同时通过阈值化把结果保持为干净的二值掩膜。

### 可分离高斯
三维高斯核可因式分解：

\[ G(x,y,z) = g(x)\,g(y)\,g(z), \qquad g(d) = \exp\!\left(-\frac{d^2}{2\sigma^2}\right) \]

因此三维卷积可化为三次一维卷积。

### 核构造（主机端，一次性上传）
- 半径 `kr = max(1, ⌈3σ⌉)`，长度 `klen = 2·kr + 1`；
- 权重 `w[d] = exp(-d²/(2σ²))`，归一化使 `Σ w = 1`。

### 核函数序列
```
toFloatKernel : f1[i] = M[i] ? 1.0 : 0.0                       # uint8 → float
gaussAxis X   : f1 → f2                                         # 沿 x 一维卷积
gaussAxis Y   : f2 → f1
gaussAxis Z   : f1 → f2
thresholdKernel: M[i] = (f2[i] >= threshold) ? 1 : 0            # 默认阈值 0.5
```
`gaussAxisKernel` 以复制填充累加 `Σ_d in[clamp(coord+d)] · kern[d+kr]`。计算在 `float`
下进行以避免量化误差，最后再二值化。

### 为什么能去噪却保住脏器
- 一个**孤立体素**经可分离滤波后，其峰值约为 `w(0)³ ≈ 0.023`（σ=1）——远低于 0.5，
  于是消失。
- 脏器**内部**在核范围内被前景包围，平滑后仍接近 1，得以保留。
- 在 **0.5** 处阈值化，对对称核而言近似体积中性；抬高阈值会使掩膜收缩，降低则使其膨胀。

### 开销
- 时间：`O(N · klen · 3)`。
- 显存：两块 float 缓冲 `f1, f2` 共 `8·N` 字节，外加很小的核数组。
- 若 `sigma ≤ 0`，本阶段跳过。

---

## 阶段 5 —— 收尾：体积统计与输出

**文件：** [`main.cu`](../src/main.cu)（配合 [`volume.cu`](../src/volume.cu) 中的
`countForegroundDevice`）

### 步骤
1. **D2H** 拷贝 `d_mask → h_out`。
2. **在 GPU 上统计前景数**（`countForegroundDevice`）：一个归约核对每个前景体素执行
   `atomicAdd(&counter, 1)`（counter 为 `unsigned long long`）——避免在主机端对 `N`
   做慢速循环。
3. **体积统计**：

   \[ V_{\text{mm}^3} = \text{finalFg} \cdot dx \cdot dy \cdot dz, \quad
      V_{\text{cm}^3} = V_{\text{mm}^3}/1000, \quad
      V_{\text{L}} = V_{\text{mm}^3}/10^{6} \]

4. **写出结果**：
   - `<prefix>_final.u8` —— 清理后的二值掩膜（原始 `uint8`，`N` 字节），可结合
     `meta.txt` 的尺寸+间距直接载入 3D Slicer / ITK-SNAP / ParaView 做**三维可视化**；
   - `<prefix>_preview_z###.pgm` —— 在 `z = p·(D-1)/(np-1)` 处均匀分布的轴向预览图。
5. **可选评估**：与金标准堆栈计算 Dice 系数
   \( \text{Dice} = \frac{2|A \cap B|}{|A| + |B|} \)，分别对重建掩膜和最终掩膜报告
   （仅用于评估，不属于流水线本身）。

---

## 计时方法

**计时器：** `Stopwatch`（[`common.cuh`](../src/common.cuh)）使用
`std::chrono::high_resolution_clock`，并在读取停止时间前调用 `cudaDeviceSynchronize()`，
因此每次测量都是真实的墙钟时间，包含核启动、设备执行以及临时缓冲的
`cudaMalloc`/`cudaFree`。

**让数据可信：**
- **预热（warm-up）** —— 先不计时地跑一遍完整链路，让 GPU 从空闲功耗状态升频，并预热
  缓存分配器（否则首次运行会被升频与首次触碰分配所主导）。
- **重复取均值** —— GPU 阶段运行 `--repeat R` 次（默认 5）后取平均。每次重复之间从一份
  原始设备副本 `d_recon` 复位体数据（一次不计时的 D2D 拷贝），使每次重复都从相同状态开始。
- **不计入**阶段耗时的部分：预热、复位拷贝、主机端金标准加载、结果打印。

### 实测结果 —— RTX 5070 Ti，256×256×128（8.39M 体素），5 次均值

| 步骤 | 耗时 | 备注 |
|---|---:|---|
| 1a. 加载切片（主机） | ~0.93 ms | 文件 I/O |
| 1b. H2D + 二值化 | ~0.69 ms | PCIe 上传 + `O(N)` 核 |
| 2. 三维 CCL + 筛选 | ~1.22 ms | 174 → 1 个连通域，3 次迭代 |
| 3. 形态学（闭+开） | ~0.77 ms | 12 遍可分离滤波 |
| 4. 高斯 + 阈值化 | ~0.82 ms | σ=1.0 |
| 5. 收尾（D2H+统计+写盘） | ~8.68 ms | **受 I/O 支配**（写 8 MB） |
| **合计** | **~13.1 ms** | |

纯 GPU 计算（1b–4）只有几毫秒；最后一步主要耗在把掩膜写盘上。在合成的类 UNet 预测上，
质量指标为：**Dice 0.9845 → 0.9876（+0.0031）**，脏器体积约 **1777.7 cm³**。

---

## 正确性验证

**核函数自测**（`./bin/liver3d --selftest`，[`selftest.cu`](../src/selftest.cu)）使用
带有手算答案的极小体数据：
- **CCL：** 一个 4×4×4 方块（64 体素）+ 一个独立的 2 体素斑点 → `found=2`、`kept=1`
  （minSize=10），斑点被清除、方块被保留。
- **闭运算：** 两块被空切片 `z=3` 隔开的薄层 → 间隙体素被填充（层间桥接）。
- **开运算：** 一个带 1 体素细尖突的方块 → 尖突被去除、方块保留（印证上文对三维盒状
  结构元的要求）。
- **高斯：** 一个 8³ 方块 + 一个孤立体素 → 方块内部保留、孤立体素被抑制。

**集成测试**（[`../tests/test_pipeline.py`](../tests/test_pipeline.py)）运行可执行程序
并断言：所有步骤都报告了耗时、CCL 降低了连通域数量、输出严格为二值、且其体素数/体积/
Dice 与报告一致、精度没有回退。

---

## 参数调优指南（肝脏掩膜）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--bin` | 0 | 输入前景阈值；`0/1` 掩膜保持 0，`0/255` 用 127 |
| `--minsize` | 64 | 删除小于此体积的连通域。肝脏约 7·10⁵ 体素，假阳性 ≪ 100 → 64 很安全；想更激进可调大 |
| `--keep-largest` | 关 | 只保留最大的单个连通域（最严格的假阳性去除） |
| `--mrx/--mry` | 1 | 面内平滑 / 去毛刺 |
| `--mrz` | 1 | **调大可弥合更厚的层间间隙**（2 → 两张切片的间隙） |
| `--no-close`/`--no-open` | 关 | 关闭某一个形态学方向 |
| `--sigma` | 1.0 | 边缘平滑强度（常用 0.8–1.5） |
| `--gthr` | 0.5 | 重新二值化的阈值；>0.5 收缩、<0.5 膨胀 |

---

## 复杂度与显存汇总

| 阶段 | 时间 | 设备显存 | 核启动次数 |
|---|---|---|---|
| 1 重建 | `O(N)` + I/O | `2·N`（临时） | 1 |
| 2 CCL + 筛选 | `O(K·N·13) + O(N)` | `8·N`（parent+counts） | `2K + 3` |
| 3 形态学 | `O(N·(rx+ry+rz))` | `2·N`（临时） | 12 |
| 4 高斯 | `O(N·klen·3)` | `8·N`（两块 float） | 5 |
| 5 收尾 | `O(N)` + I/O | 很小（计数器） | 1 |

（`N` 字节 = 每体素 1 字节；`4·N` = `int`；`8·N` = `float`/两块缓冲。对 8.39M 体素：
`N ≈ 8 MB`，`8·N ≈ 67 MB`。）

---

## 扩展方向
- **面向三维可视化的网格导出：** 在 `d_mask` 上增加一个移动立方体（marching cubes）
  阶段，输出 STL/PLY/OBJ，作为原始体数据的替代或补充。
- **接入真实 UNet 结果：** 用 [`../tools/import_masks.py`](../tools/import_masks.py)
  把预测转换（PNG/PGM/PBM/RAW → `slices/*.u8` + `meta.txt`）。公开肝脏 CT 数据源：
  **3D-IRCADb-01**、**LiTS**。
- **有符号距离/浮点掩膜：** 修改 `binarizeKernel` 的阈值逻辑，即可直接摄入概率图。
