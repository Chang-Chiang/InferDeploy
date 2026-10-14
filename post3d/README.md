# 3D-Process —— 基于 CUDA 的三维脏器掩膜后处理流水线

把**逐层分割掩膜**（例如对每张轴向 CT 切片运行的 2D UNet 结果）重建为**三维体数据**，
再用一条完全基于 **CUDA** 的后处理流水线对其进行清理，最终输出规整的脏器掩膜，
用于**体积统计**与**三维可视化**。

> 任务：*将逐层分割掩膜重建为三维体数据，实现基于 CUDA 的后处理流水线——最大连通域标记与筛选去除层间孤立假阳性区域，3D 形态学开闭弥合层间不连续并消除孔洞毛刺，3D 高斯平滑抑制边缘噪声，输出规整的脏器三维掩膜，支撑后续体积统计与三维可视化。*

开发与测试平台：**NVIDIA RTX 5070 Ti（Blackwell 架构，cc 12.0）**。

> **算法详细讲解：** [`docs/ALGORITHM.zh-CN.md`](docs/ALGORITHM.zh-CN.md)——完整的处理流程
> 与每一个算法（并行并查集连通域标记、可分离 box 形态学、可分离高斯），含伪代码、
> 复杂度分析、正确性论证、计时方法与基准数据。
> （英文版：[`docs/ALGORITHM.md`](docs/ALGORITHM.md)）

---

## 处理流程

```
 逐层 2D 掩膜 (slice_%04d.u8)
        │
   [1] 三维重建    ── 主机端拼接 + H2D 上传 + 二值化   →  uint8 体数据 (W×H×D)
        │
   [2] 3D CCL+筛选 ── GPU 并查集，26 邻接             →  丢弃 < minSize 的连通域
        │                                              （去除层间孤立假阳性区域）
   [3] 3D 形态学   ── 可分离 box 闭运算后开运算       →  弥合层间缝隙、消除孔洞与毛刺
        │
   [4] 3D 高斯平滑 ── 可分离 3D 高斯 + 0.5 阈值       →  抑制边缘噪声、平滑表面
        │
   [5] 收尾输出    ── D2H、体积统计、写出掩膜与 PGM   →  规整的脏器三维掩膜
```

每个阶段都用挂钟秒表计时，并在计时前同步设备；GPU 阶段在预热一遍之后重复
`--repeat` 次取平均值，因此结果不会被冷启动的时钟爬升或首次显存分配开销所干扰。

---

## 目录结构

```
Makefile                  CUDA 构建（默认产出 sm_120 原生 SASS，无需驱动 JIT）
run.sh                    构建 → 自测 → 生成数据 → 运行 → 校验
src/
  common.cuh              CUDA 错误检查、grid 辅助函数、Stopwatch
  volume.cuh / volume.cu  [1] 切片→三维重建、raw/PGM 读写、体素计数
  ccl.cuh    / ccl.cu     [2] 三维连通域标记（并查集）+ 体积筛选
  morphology.cuh/.cu      [3] 可分离 3D box 腐蚀/膨胀 → 开/闭运算
  gaussian.cuh/.cu        [4] 可分离 3D 高斯平滑 + 阈值
  selftest.cuh/.cu        阶段 2–4 的已知答案单元测试
  main.cu                 [5] CLI、流程编排、逐步骤计时报告
tools/
  gen_test_data.py        合成 UNet 风格的肝脏掩膜（仅依赖 numpy）
  import_masks.py         导入真实 UNet 掩膜（PNG/PGM/PBM/RAW）为流水线输入
tests/
  test_pipeline.py        端到端集成测试（解析报告 + numpy 校验）
reference/
  union_find.cpp          纯 C++ 并查集 + CPU 三维连通域标记参考实现（对应 src/ccl.cu）
  ptx_jit_demo.cu         PTX/SASS/JIT 最小可运行样例（`make ptxdemo`，驱动 JIT 能力冒烟测试）
docs/
  ALGORITHM.md            detailed algorithm flow + per-algorithm explanation (EN)
  ALGORITHM.zh-CN.md      算法流程与每个算法的详细讲解（中文）
  CUDA_GUIDE.zh-CN.md     以本项目为例的 CUDA 知识详解（中文）
  PTX_JIT_SASS.zh-CN.md   PTX/SASS/JIT 与 `-gencode` 逐字段详解（中文，含实测输出）
  UNION_FIND.zh-CN.md     并查集详解 + 纯 C++ 实现讲解（中文）
data/                     生成的输入/输出（已被 git 忽略）
```

---

## 环境要求

- **CUDA toolkit**（`nvcc`）与一块 NVIDIA GPU。本机实测为 **CUDA 13.4**；若要为
  RTX 50 系列原生编译 `sm_120`，需 **CUDA ≥ 12.8**。
- 支持 **C++17** 的主机编译器，GNU `make`。
- **Python 3** 与 **numpy**（用于生成测试数据 / 导入掩膜 / 测试）。Pillow 为可选依赖
  （仅在导入 PNG/JPG/TIFF 掩膜时需要）。

### GPU 架构说明（Blackwell / sm_120）
本机为 **CUDA 13.4** + 驱动 **595.99.02**，GPU 为 RTX 5070 Ti（`sm_120`，cc 12.0）；
`nvcc` 已直接支持 `compute_120`，因此 `Makefile` 默认产出**原生 SASS**
（`-gencode arch=compute_120,code=sm_120`），运行时不需要驱动 JIT。

> ⚠️ **不要退回「纯 PTX + 驱动 JIT」的老写法**（`arch=compute_90,code=compute_90`）。
> 那是 CUDA 12.4 时代的妥协（当时 `nvcc` 不认识 `sm_120`）。CUDA 13.x 生成的 PTX 为
> **ISA 9.4**，比本驱动（上限 CUDA 13.2）更新，驱动会拒绝 JIT 并报
> `the provided PTX was compiled with an unsupported toolchain`，`--selftest` 随即失败。
> 经验：**工具链比驱动新时，必须嵌 SASS**。仓库里旧的二进制（CUDA 12.4 构建，PTX ISA 8.4）
> 仍可被 JIT 跑起来，所以「旧产物能跑、新构建跑不了」并不矛盾。

> 另注：CUDA 13 已移除 `compute_61`（Pascal）等旧架构，`nvcc --list-gpu-arch` 最低只到
> `sm_75`；给旧工具链写的 `-gencode` 在新版上会直接报 `Unsupported gpu architecture`。
>
> PTX / SASS / 驱动 JIT 的完整原理、`.version` 与驱动上限的关系、`-gencode` 逐字段与
> 排查命令，见 [`docs/PTX_JIT_SASS.zh-CN.md`](docs/PTX_JIT_SASS.zh-CN.md)（配套可运行样例
> `make ptxdemo`）。

若要面向其他 GPU，可覆盖架构参数，例如：

```bash
make GENCODE="-gencode arch=compute_86,code=sm_86"     # RTX 30 系列
make GENCODE="-gencode arch=compute_89,code=sm_89"     # RTX 40 / L40
make GENCODE="-gencode arch=compute_120,code=[sm_120,compute_120]"   # SASS + PTX（仅驱动不匹配时会用到 PTX 分支）
```

---

## 快速开始

```bash
./run.sh                    # 构建 + 自测 + 生成数据 + 运行 + 校验
./run.sh --minsize 128 --mrz 2 --sigma 1.5 --repeat 10   # 参数会转发给可执行文件
REGEN=1 ./run.sh            # 强制重新生成测试数据
```

手动执行：

```bash
make                                   # -> bin/liver3d
python3 tools/gen_test_data.py         # -> data/slices/*.u8, data/meta.txt, data/liver_gt.u8
./bin/liver3d --selftest               # 核函数单元测试
./bin/liver3d                          # 运行流水线并打印计时
python3 tests/test_pipeline.py         # 集成校验
```

---

## 命令行选项

| 选项                       | 默认值             | 含义                                               |
| -------------------------- | ------------------ | -------------------------------------------------- |
| `--slices <dir>`           | `data/slices`      | 逐层原始掩膜目录 `slice_%04d.u8`（每个 W·H 字节）  |
| `--meta <file>`            | `data/meta.txt`    | `W H D dx dy dz`（尺寸 + 体素间距，单位毫米）      |
| `--stack <file>`           | –                  | 单个连续原始体数据文件（覆盖 `--slices`）          |
| `--gt <file>`              | `data/liver_gt.u8` | 用于计算 Dice 的真值体数据                         |
| `--no-gt`                  | –                  | 跳过真值 / Dice 评估                               |
| `--out <prefix>`           | `data/out/liver`   | 输出前缀（`_final.u8`、`_preview_z*.pgm`）         |
| `--bin <int>`              | `0`                | 二值化阈值（`raw > bin` 处为前景）                 |
| `--minsize <int>`          | `64`               | CCL：移除小于该体素数的连通域                      |
| `--keep-largest`           | 关闭               | CCL：只保留最大的那一个连通域                      |
| `--mrx/--mry/--mrz <int>`  | `1 1 1`            | 形态学 box 半径（体素；增大 `mrz` 可弥合层间缝隙） |
| `--no-close` / `--no-open` | 关闭               | 禁用形态学闭运算 / 开运算                          |
| `--sigma <float>`          | `1.0`              | 高斯 sigma（体素）                                 |
| `--gthr <float>`           | `0.5`              | 高斯后的重新二值化阈值                             |
| `--previews <int>`         | `3`                | 轴向 PGM 预览图数量                                |
| `--repeat <int>`           | `5`                | 每个 GPU 阶段的计时重复次数（取平均）              |
| `--selftest`               | –                  | 运行 CUDA 核函数自测后退出                         |

---

## 测试数据

### 合成数据（默认，可复现）
`tools/gen_test_data.py` 会构造一个类肝脏的真值团块（两个椭球叶 + 心切迹，
在 1×1×2.5 mm 间距下约 1776 cm³），以及一个模拟典型逐层 UNet 缺陷的**预测**掩膜：

- 边界抖动 / 锯齿（逐层推理噪声）
- **层间不连续**（切片错位 + 丢层 → 细小缝隙）
- 内部孔洞（血管被误标为背景）
- **边缘毛刺**（表面细小凸起）
- **孤立假阳性**（仅存在于 1–2 层的小团块）

输出 `data/slices/slice_*.u8`（流水线输入）、`data/meta.txt`、
`data/liver_pred_stack.u8`、`data/liver_gt.u8` 以及 `data/preview/*.pgm`。

### 真实 UNet 肝脏结果
在公开肝脏 CT 数据集上运行你的 UNet，再导入其逐层预测结果：

```bash
python3 tools/import_masks.py --indir unet_pred_png --out data \
    --ext '*.png' --dx 0.7 --dy 0.7 --dz 2.0 --threshold 127
./bin/liver3d --no-gt      # 若同时导入了真值，也可保留 --gt
```

`import_masks.py` 支持 **PNG/JPG/TIFF**（依赖 Pillow）、**PGM/PBM**（纯 numpy）
以及 **RAW** 字节流。常用的公开肝脏 CT / 分割掩膜数据源：

- **3D-IRCADb-01** — 20 例腹部 CT，含肝脏（及肿瘤）掩膜。<https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/>
- **LiTS**（肝脏肿瘤分割挑战赛）— 130+ 例增强 CT。<https://competitions.codalab.org/competitions/17094>

---

## 算法

**[2] 三维连通域标记（`src/ccl.cu`）。** 基于 26 邻接的并行**并查集**。每个体素初始
各自为根；*merge* 核函数用 `atomicMin` 把每个前景体素与其 13 个前向邻域合并到同一个根
（根只可能变小 ⇒ 保证终止），*compress* 核函数做指针跳跃。两者交替执行直到没有父节点
发生变化（通常 2–4 轮）。随后用逐根计数直方图（`atomicAdd`）驱动筛选：体素数少于
`minSize` 的连通域被清零——这就是去除层间孤立假阳性的一步。同时报告连通域个数、
最大连通域体积，以及筛选前后的前景体素总数。

**[3] 三维形态学（`src/morphology.cu`）。** 使用 **box** 结构元素的二值腐蚀/膨胀，
实现为**三次可分离的一维 min/max 扫描**（X、Y、Z）——每个体素 O(r) 而非 O(r³)。
*闭运算*（先膨胀后腐蚀）弥合层间缝隙并填充孔洞；*开运算*（先腐蚀后膨胀）去除毛刺与
细小尖刺。分轴半径可以沿 Z 方向拉长结构元素，专门用于修补层间不连续。体数据边界采用
**复制（clamp）填充**。

**[4] 三维高斯平滑（`src/gaussian.cu`）。** 把掩膜转为 float，沿 X、Y、Z 与**可分离**的
一维高斯核卷积（核半径 `⌈3σ⌉`，权重归一化），再按 `--gthr`（默认 0.5）重新阈值化。
这一步抑制高频边缘噪声，在近似保持体积的前提下得到平滑规整的表面。

---

## 基准数据（RTX 5070 Ti，256×256×128 = 839 万体素，GPU 阶段取 5 次平均）

环境：**CUDA 13.4 + 驱动 595.99.02**，默认原生 `sm_120` SASS 构建（2026-10-12 实测）：

```
 1a. 三维重建（读取切片）              ~1.10 ms     （主机端文件 I/O）
 1b. 三维重建（H2D + 二值化）          ~0.61 ms
 2.  3D CCL + 连通域筛选               ~1.30 ms     （174 → 1 个连通域）
 3.  3D 形态学（闭 + 开）              ~0.78 ms
 4.  3D 高斯平滑 + 阈值                ~0.64 ms
 5.  收尾输出（D2H + 统计 + 写文件）    ~11.0 ms     （主要是 8 MB 文件写入）
 --------------------------------------------------------------
 总计                                 ~15.4 ms
```

> 历史对照：CUDA 12.4 + `compute_90` PTX（驱动 JIT）时测得总计 ~14.6 ms（1b ~0.85、3 ~1.00、
> 5 ~9.6 ms）。两者在同一量级，差异主要是主机文件 I/O 与时钟状态的波动；**质量指标完全一致**
> （下述 Dice 与体积逐项复现）。

对质量的影响（合成的类 UNet 预测结果 vs. 真值）：

```
 Dice  重建后 0.9845  →  最终 0.9876   （+0.0031）
 脏器体积 ≈ 1777.7 cm³ (1.78 L)
```

纯 GPU 计算（步骤 1b–4）对 839 万体素的体数据只需几毫秒（实测合计 ~3.3 ms）；步骤 5 是 I/O 瓶颈
（把掩膜与预览图写入磁盘）。

---

## 输出结果

- `data/out/liver_final.u8` —— 清理后的二值掩膜（W·H·D 字节，取值 0/1），可直接导入
  **3D Slicer / ITK-SNAP / ParaView**（按 `meta.txt` 中的尺寸与间距作为 raw uint8 加载），
  用于三维可视化与后续分析。
- `data/out/liver_preview_z###.pgm` —— 轴向 PGM 预览图，便于快速查看。
- 控制台报告 —— 各步骤耗时、连通域统计、前景体素数、**脏器体积**（mm³ / cm³ / L）
  以及与真值的 Dice。

---

## 校验

- `./bin/liver3d --selftest` —— 已知答案单元测试：CCL 保留大块并丢弃小碎块；闭运算
  弥合 1 层的 Z 向缝隙；开运算去除细小尖刺；高斯保留实心块并抹去孤立体素。
- `python3 tests/test_pipeline.py` —— 运行可执行文件后校验：所有步骤都报告了耗时、
  CCL 减少了连通域数量、输出严格为二值且体素数/体积/Dice 一致、精度没有下降。
