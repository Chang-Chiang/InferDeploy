# CUDA 知识文档 —— 以“三维脏器掩膜后处理流水线”为例

本文以本项目 [`../src`](../src) 中的真实核函数为线索，系统梳理其中用到的每一项 CUDA
知识：从编程模型、内存管理、原子操作，到并查集/不动点迭代、可分离滤波、性能计时与
体系结构编译。每个概念都对应到具体代码，便于“照着代码学 CUDA”。
（算法本身的详解见 [`ALGORITHM.zh-CN.md`](ALGORITHM.zh-CN.md)。）

---

## 0. 概念 → 代码 速查表

| CUDA 概念 | 本项目中的体现 | 文件 |
|---|---|---|
| 一维核 / 全局线程 id | `binarizeKernel`、`thresholdKernel`、`countFgKernel` | `volume.cu`、`gaussian.cu` |
| 三维核 / `dim3` 网格 | `mergeKernel`、`morphAxisKernel`、`gaussAxisKernel` | `ccl.cu`、`morphology.cu`、`gaussian.cu` |
| 全局内存管理 | `cudaMalloc/Free/Memcpy/Memset` | 全部 |
| `__restrict__` 无别名 | 所有读写指针参数 | 全部 |
| 原子直方图 | `atomicAdd(&counts[...], 1)` | `ccl.cu` |
| 原子归约计数 | `atomicAdd(counter, 1)`（`unsigned long long`） | `volume.cu` |
| `atomicMin` 并查集 union | `atomicMin(&parent[hi], lo)` | `ccl.cu` |
| `atomicMax` 打包取最大 | `(count<<32)|label` | `ccl.cu` |
| 不动点迭代 + changed 标志 | merge/compress 收敛循环 | `ccl.cu` |
| 可分离滤波 + 乒乓缓冲 | 3×1D min/max、3×1D 卷积 | `morphology.cu`、`gaussian.cu` |
| 同步与错误检查 | `CUDA_CHECK`、`cudaDeviceSynchronize` | `common.cuh` |
| 墙钟计时 | `Stopwatch` | `common.cuh` |
| PTX / 架构编译 | `-gencode arch=compute_90,code=compute_90` | `Makefile` |

---

## 1. CUDA 编程模型基础

### 1.1 主机(host)与设备(device)
CUDA 程序由两部分组成：CPU（host）负责流程控制、内存分配与数据搬运；GPU（device）
负责大规模并行计算。本项目中 host 侧在 [`main.cu`](../src/main.cu) 里按阶段调用
`filterComponents()` / `morphOpenClose()` / `gaussianSmoothMask()` 等函数，这些函数内部
再启动 GPU 核。

### 1.2 三种函数限定符
```cuda
__global__ void mergeKernel(...) { ... }          // 核函数：host 调用，device 执行，必须返回 void
__device__ __forceinline__ int uf_find(...) {...}  // 设备函数：device 调用，device 执行
```
- `__global__`：核函数（kernel），用 `<<<...>>>` 启动，返回类型必须是 `void`。
- `__device__`：只能在设备上被调用/执行的函数，常用于封装小工具（如 `uf_find`、`clampi`）。
- `__forceinline__`：强制内联，消除函数调用开销——适合 `uf_find`、`clampi` 这类高频小函数。

### 1.3 线程层级：grid / block / thread
GPU 以“网格(grid) → 线程块(block) → 线程(thread)”三级组织并行。每个线程用内建变量定位自己：
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;   // 全局 x 坐标
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```
一维核则简化为 `size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;`
（注意用 `size_t` 防止 `N` 超过 `int` 范围而溢出——本项目 `N≈8.4M` 虽未越界，但这是好习惯）。

### 1.4 一维 vs 三维网格：`dim3`
本项目两种启动方式并存：
- **三维核**（处理体数据）用 `dim3 block(8,8,8)`（=512 线程/块），网格由
  [`common.cuh`](../src/common.cuh) 的 `gridFor()` 按 `(W,H,D)` 向上取整计算：
  ```cuda
  inline dim3 gridFor(int W,int H,int D, dim3 block = dim3(8,8,8)) {
      return dim3((W+block.x-1)/block.x, (H+block.y-1)/block.y, (D+block.z-1)/block.z);
  }
  ```
- **一维核**（线性遍历 `N`）用 `threads=256`、`blocks=(N+255)/256`。

> 经验：每块线程数应是 32（warp 大小）的倍数，常见 128/256/512；上限 1024。`8×8×8=512`
> 对三维 stencil 类核很合适。

### 1.5 核启动语法与边界判断
```cuda
morphAxisKernel<<<grid, block>>>(in, out, W, H, D, axis, r, dilate);
```
`<<<grid, block>>>` 之间还可加第三/四参数（共享内存字节数、stream）。因为网格是向上取整的，
边缘块会有“多余”线程，核内必须做边界判断：
```cuda
if (x >= W || y >= H || z >= D) return;   // 三维核
if (i >= N) return;                       // 一维核
```
漏掉这一步会造成越界读写，属于最常见的新手错误。

---

## 2. 三维体数据的索引与访存

### 2.1 行主序布局
体数据按 `i = (z*H + y)*W + x` 线性存放（z 变化最慢，x 最快）。这样“同一 z、同一 y、
x 连续”的体素在内存中也连续。

### 2.2 全局内存合并访问(coalescing)
GPU 全局内存以 32/128 字节事务读取。当一个 warp 内相邻线程访问相邻地址时，多次访问会被
**合并**成少数几次事务，带宽利用率最高。由于 x 是最快维，让 `threadIdx.x` 对应 x 坐标
（`dim3 block(8,8,8)` 中 x 最先变化），相邻线程正好读相邻 `x` → 合并访问。这是三维核仍用
`(x,y,z)` 顺序而非 `(z,y,x)` 的关键原因。

### 2.3 边界处理：复制填充(clamp)
stencil 类核（形态学、高斯）在边界处需要访问邻居。本项目用 `clampi` 做**复制填充**：
```cuda
__device__ __forceinline__ int clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
nx = clampi(x + d, 0, W - 1);
```
好处：无需额外的填充缓冲区，且不会因“隐式背景”而在脏器边缘产生人为腐蚀/膨胀。

### 2.4 `__restrict__` 与别名
```cuda
__global__ void gaussAxisKernel(const float* __restrict__ in, float* __restrict__ out, ...)
```
`__restrict__` 向编译器承诺 `in`、`out` 不重叠（无别名），从而允许更激进的加载/缓存与
指令重排优化。对只读输入再加 `const`，语义更清晰。

> 注意：使用 `__restrict__` 的前提是调用方确实不传重叠指针。本项目 `separable()` 严格保证
> `dst != B`（见第 7 节），正是为满足这一前提。

---

## 3. 内存管理

### 3.1 分配/释放/拷贝/置零
```cuda
CUDA_CHECK(cudaMalloc(&parent, N * sizeof(int)));            // 分配设备内存
CUDA_CHECK(cudaMemset(counts, 0, N * sizeof(int)));          // 置零
CUDA_CHECK(cudaMemcpy(d_raw, h_raw, n, cudaMemcpyHostToDevice)); // H2D
CUDA_CHECK(cudaMemcpy(h_out.data(), d_mask, N, cudaMemcpyDeviceToHost)); // D2H
CUDA_CHECK(cudaMemcpy(d_mask, d_recon, N, cudaMemcpyDeviceToDevice));    // D2D（复位用）
CUDA_CHECK(cudaFree(parent));                                // 释放
```
四种拷贝方向由枚举指定：`cudaMemcpyHostToDevice / DeviceToHost / DeviceToDevice /
HostToHost`。本项目用 **D2D** 在计时重复之间把体数据复位到“原始重建态”，代价远小于走主机。

### 3.2 数据类型选择
| 用途 | 类型 | 原因 |
|---|---|---|
| 二值掩膜 | `uint8` | 1 字节/体素，省带宽与显存 |
| 卷积中间量 | `float` | 避免整数量化，保证高斯权重精度 |
| 并查集 parent / counts | `int` | 索引与计数，`int` 足够且原子支持好 |
| 大计数归约 | `unsigned long long` | 防止溢出，且 **CUDA 有该类型的 `atomicAdd`**（见 §5.4） |

### 3.3 临时/乒乓缓冲
- 形态学用 `t1,t2` 两块 `uint8` 缓冲做三遍可分离滤波的乒乓倒换。
- 高斯用 `f1,f2` 两块 `float` 缓冲。
这些缓冲在函数内 `cudaMalloc` / `cudaFree`，作用域清晰。

### 3.4 缓存分配器与首次开销
CUDA 运行时内部有**缓存分配器**：`cudaFree` 的显存常被缓存以便下次 `cudaMalloc` 快速复用。
但**首次**分配、以及 GPU 从空闲功耗态升频，会让“第一次运行”明显偏慢。这正是本项目在计时前
先跑一遍 **warmup**（[`main.cu`](../src/main.cu)）的原因（详见 §9.5）。

### 3.5 托管内存(Managed Memory)
最小示例里可用 `cudaMallocManaged` 让 host/device 共享同一指针、由驱动按需迁移页：
```cuda
int* a; cudaMallocManaged(&a, 4); k<<<1,1>>>(a); cudaDeviceSynchronize();
```
本项目为精确控制传输与计时，采用显式 `cudaMalloc + cudaMemcpy`，而非托管内存。

---

## 4. 同步与错误处理

### 4.1 核启动是异步的
`kernel<<<...>>>()` 立即返回，GPU 在后台执行。若不同步就读结果或计时，会得到错误数据/时间。
需要 `cudaDeviceSynchronize()` 阻塞到所有设备工作完成。

### 4.2 两类错误、两个 API
- **启动错误**（配置非法、参数错误）：由 `cudaGetLastError()` 捕获。
- **执行错误**（核内非法访存等）：通常在后续同步或 `cudaPeekAtLastError()` 时暴露。
本项目在每个可能出错的调用后用 `CUDA_CHECK` 包裹：
```cuda
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err__ = (call);                                        \
    if (err__ != cudaSuccess) {                                        \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  -> %s\n",           \
                __FILE__, __LINE__, #call, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE);                                            \
    } } while(0)
```
把 `cudaGetErrorString(err__)` 打印出来，是定位 CUDA 问题的第一手段。

### 4.3 计时为何要同步
[`common.cuh`](../src/common.cuh) 的 `Stopwatch::stopMs()` 在读时钟前先
`cudaDeviceSynchronize()`，确保测到的是“真正跑完”的墙钟时间，而非仅“启动完成”的时间。

---

## 5. 原子操作(Atomics)

### 5.1 为什么需要原子
多个线程同时“读-改-写”同一地址会产生**竞态(race)**，结果不确定。直方图统计、全局计数、
并查集合并都属于此类，必须用原子指令保证读-改-写不可分割。

### 5.2 `atomicAdd`：直方图与归约
```cuda
// 直方图：同一连通域的体素累加到同一槽位
atomicAdd(&counts[ uf_find(parent, i) ], 1);
// 全局前景计数（归约）
if (i < n && mask[i]) atomicAdd(c, 1ULL);   // c 是 unsigned long long*
```

### 5.3 `atomicMin` / `atomicMax`
```cuda
atomicMin(&parent[hi], lo);        // 并查集 union：把较大的根挂到较小的根
atomicMax(&s->packedMax, key);     // 取“最大连通域”
```
`atomicMin` 保证 `parent[hi]` 只会变小——这是并查集迭代**必然收敛**的关键（见 §6.2）。

### 5.4 支持的类型与一个真实的坑
CUDA 的 `atomicAdd` **没有 `long long*`（有符号）重载**，只有 `int*`、`unsigned int*`、
`float*`、`double*`、`unsigned long long*`。本项目最初写成 `atomicAdd(&keptFg,(long long)c)`
编译报错，改为 `unsigned long long` 后通过。
> 记住：**64 位原子累加用 `unsigned long long`。**

### 5.5 打包技巧：一次原子取“最大分量及其标签”
需要同时知道“最大连通域的体积”和“它是哪个（标签）”。把两者打包进一个 64 位整数：
```cuda
unsigned long long key = ((unsigned long long)(unsigned)c << 32) | (unsigned)i; // 高32位=count, 低32位=label
atomicMax(&packedMax, key);
```
因为 count 在高位，`atomicMax` 会先按 count 比较，选出最大分量；随后：
```cuda
int largestLabel = (int)(packedMax & 0xffffffffull);
int largestCount = (int)(packedMax >> 32);
```
一次原子操作完成“二维”归约，避免两遍扫描。

### 5.6 原子性能
高冲突（大量线程打同一地址）会让原子串行化、性能退化。本项目直方图的冲突较低（标签分散）。
若冲突高，可考虑：共享内存做块内局部归约再合并、或分桶(slicing)降低争用（见 §13）。

---

## 6. 并行并查集与不动点迭代（GPU 上的图算法）

这是本项目最有“CUDA 特色”的部分：把串行的连通域标记改写成可并行的不动点迭代。

### 6.1 find 只读，避免写竞争
```cuda
__device__ __forceinline__ int uf_find(const int* parent, int x) {
    int r = x; while (parent[r] != r) r = parent[r]; return r;
}
```
`uf_find` **只读**地走到根，不在其中做路径压缩——因为 merge 阶段成千上万线程并发调用它，
若在里面写 `parent` 会互相竞争。压缩交给独立的 `compressKernel` 完成。

### 6.2 merge：单调递减保证终止
```cuda
int lo = min(ri, rj), hi = max(ri, rj);
atomicMin(&parent[hi], lo);   // 根标签只减不增
```
每次 union 都把某个根指向**更小**的索引，标签集合有下界（0），因此迭代**必然终止**
（Noetherian）。这是并发并查集正确收敛的核心论证。

### 6.3 compress：指针跳跃
```cuda
int r = uf_find(parent, i);
if (parent[i] != r) { parent[i] = r; *changed = 1; }
```
把每个节点直接挂到根，路径长度每轮大致减半，使 merge 中的 `find` 越来越快、迭代次数很少
（本项目实测 2–4 次）。

### 6.4 changed 标志 + 主机-设备往返的收敛循环
```cuda
while (true) {
    int zero = 0; cudaMemcpy(d_changed, &zero, 4, cudaMemcpyHostToDevice); // 复位
    mergeKernel<<<grid,block>>>(d_mask, parent, W,H,D, d_changed);
    compressKernel<<<blocks1d,threads>>>(parent, N, d_changed);
    int ch = 0; cudaMemcpy(&ch, d_changed, 4, cudaMemcpyDeviceToHost);      // 读回
    if (!ch) break;                                                          // 不动点
    if (++iter > 500) break;                                                 // 安全上限
}
```
这是 GPU 上表达“**重复直到稳定**”的标准模式：设备端把“是否发生变化”写进一个标志，主机端
每轮同步读回判断是否继续。

### 6.5 幂等性
merge 对每条边可能被多个线程重复处理，但 `atomicMin` 是**幂等**的（重复取 min 结果不变），
因此并发与重复都安全。

### 6.6 进阶：减少主机往返
每轮两次 4 字节的 H2D/D2H 虽小，但会打断流水。可用：
- **pinned/mapped（zero-copy）内存**放 `changed` 标志，主机直接读设备内存；
- **协作组(Cooperative Groups) 的 `grid.sync()`** 在设备端做全局同步，把整个迭代留在 GPU 上
  （需协作启动 `cudaLaunchCooperativeKernel`）。

---

## 7. 可分离滤波：把 3D 拆成 3×1D

### 7.1 可分离性原理
- **盒状结构元**是三段的 Minkowski 和，min/max 滤波可分离：3D 盒腐蚀 = 沿 X、Y、Z 的三次
  1D 腐蚀。
- **高斯核**可因式分解 `G(x,y,z)=g(x)g(y)g(z)`，故 3D 卷积 = 三次 1D 卷积。

### 7.2 复杂度收益
直接 3D 邻域是 `O(r^3)`（形态学）或 `O(k^3)`（卷积）；可分离后降到 `O(r)` / `O(k)` 每体素。
对 `r=1`、`k=7` 这类小核，收益显著且实现简单。

### 7.3 乒乓缓冲与 in-place 冒险
三遍滤波必须“读一块、写另一块”，否则会在同一遍里边读边写产生错误：
```cuda
// X: src→A, Y: A→B, Z: B→dst  （要求 dst != B）
morphAxisKernel<<<grid,block>>>(src, A,   W,H,D, 0, rx, dilate);
morphAxisKernel<<<grid,block>>>(A,   B,   W,H,D, 1, ry, dilate);
morphAxisKernel<<<grid,block>>>(B,   dst, W,H,D, 2, rz, dilate);
```
`morphOpenClose` 用两块 `t1,t2` 巧妙路由，使开/闭各两次运算都无需第三块缓冲：
```cuda
// 闭：先膨胀到 t1，再腐蚀回 mask
separable(mask, t1, t2, t1,    /*dilate*/true,  ...);
separable(t1,   t2, t1, mask,  /*erode*/ false, ...);
```
关键约束是 `dst != B`（Z 遍读 B、写 dst，不能是同一块）。`dst` 可等于 `src` 或 `A`。

### 7.4 一致(uniform)分支：避免 warp 分化
```cuda
if (axis == 0)      nx = clampi(x + d, 0, W - 1);
else if (axis == 1) ny = clampi(y + d, 0, H - 1);
else                nz = clampi(z + d, 0, D - 1);
```
`axis` 对**同一次核启动的所有线程相同**，因此该分支是“一致的”，不会造成 warp 内分化
(divergence)。这是把“沿哪个轴”作为运行期参数而非模板参数时的性能保障。

---

## 8. 归约(Reduction)与统计

### 8.1 本项目做法：全局原子归约
前景计数用一次 `atomicAdd` 到单个 `unsigned long long`：
```cuda
__global__ void countFgKernel(const uint8_t* mask, size_t n, unsigned long long* c){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && mask[i]) atomicAdd(c, 1ULL);
}
```
简单、正确，适合“统计量不大、冲突可接受”的场景，并且比主机端遍历 `N` 快得多。

### 8.2 进阶：树形归约 / warp shuffle
当原子冲突成为瓶颈时，标准做法是**两级归约**：块内用共享内存做树形归约，块间再归约；
块内最后 32 个lane 用 `__shfl_down_sync` 做 warp 级归约，几乎无共享内存开销。

### 8.3 进阶：CUB / Thrust
生产代码可直接用 `cub::DeviceReduce::Sum` 或 `thrust::reduce`，它们内部就是高度优化的
分段归约，省去手写。

---

## 9. 性能分析方法论

### 9.1 访存受限 vs 计算受限
用**算术强度**（每字节访存对应的浮点/整数运算数）判断瓶颈。本项目的核几乎都是
**访存受限(memory-bound)**：每读入一个体素只做很少的运算（min/max、几次乘加、一次比较）。
因此优化重点是**减少访存量、提高合并度、复用缓存**，而非增加算力。

### 9.2 可分离为何快
把 3D 邻域的多次读，替换为 3 次 1D 扫描，显著降低每体素的访存与计算量（§7.2）。

### 9.3 占用率(occupancy)与块大小
占用率 = 活动 warp / 硬件上限。块太小→调度开销大；太大→寄存器/共享内存受限、尾效应。
本项目 1D 用 256、3D 用 `8×8×8=512`，都是常见的高占用选择。

### 9.4 `#pragma unroll` 与寄存器
```cuda
#pragma unroll
for (int k = 0; k < 13; ++k) { ... }   // 13 个前向偏移，编译期展开
```
对**编译期已知次数**的小循环展开，可减少循环控制开销、暴露更多指令级并行。但过度展开会
增加寄存器压力、降低占用率，需权衡。

### 9.5 GPU 计时的正确姿势
- **墙钟 + 同步**（本项目）：`Stopwatch` 在 `cudaDeviceSynchronize()` 后读
  `std::chrono`，测得包含启动与执行的真实耗时。
- **CUDA 事件**（另一种）：`cudaEventRecord(start)` … `cudaEventRecord(stop)` …
  `cudaEventElapsedTime(&ms,start,stop)`，直接测 GPU 时间线。
- **务必 warmup + 多次取均值**：GPU 空闲时处于低功耗频率，首次运行会被“升频”和“首次分配”
  严重拉高。本项目先跑一遍 warmup，再对每个 GPU 阶段重复 `--repeat`（默认 5）次取平均，
  重复间用 D2D 从原始副本复位。
  > 实测对比：冷启动首跑 H2D+二值化约 14.6 ms、CCL 约 30 ms；warmup 后分别降到
  > ~0.7 ms 与 ~1.2 ms。**不 warmup 的计时会误导优化方向。**

---

## 10. 编译与体系结构

### 10.1 虚拟架构 vs 真实架构
`nvcc` 的 `-gencode arch=compute_XY,code=sm_AB` 中：
- `compute_XY`：**虚拟架构**，决定生成的中间表示 **PTX** 及可用特性。
- `sm_AB`：**真实架构**，决定为哪种 GPU 生成机器码 **SASS**。
- `code=compute_XY` → 嵌入 **PTX**（可被驱动 JIT 到更新的架构）；
  `code=sm_AB` → 嵌入该架构的 **SASS**（开箱即用、无需 JIT）。
一个二进制可嵌入多份（fatbin），运行时选最匹配的。

### 10.2 PTX 前向兼容与 JIT
驱动可把 PTX **即时编译(JIT)** 成当前 GPU 的 SASS。因此“为较老虚拟架构生成的 PTX”能在
**更新**的 GPU 上运行——这就是前向兼容。

### 10.3 本项目的真实场景（Blackwell / sm_120）
开发机是 **RTX 5070 Ti（Blackwell，`sm_120`，cc 12.0）**，但工具链是 **CUDA 12.4**，其
`nvcc` **不认识 `sm_120`**（直接 `-arch=sm_120` 会报
`Value 'sm_120' is not defined`）。解决办法：编译成 **`compute_90` 的 PTX**，交给更新的驱动
（本机 595.x，支持 CUDA 13.x）在运行时 JIT 到 `sm_120`：
```make
GENCODE ?= -gencode arch=compute_90,code=compute_90
```
已验证可正确运行。

### 10.4 为不同 GPU 覆盖架构
```bash
make GENCODE="-gencode arch=compute_86,code=sm_86"   # RTX 30 系
make GENCODE="-gencode arch=compute_89,code=sm_89"   # RTX 40 / L40
```
> 经验：**先确认 nvcc 版本是否支持目标 GPU 架构**；若不支持，退而求其次用较老的
> `compute_` PTX 让驱动 JIT，是快速可行的兼容方案。

---

## 11. 常见坑与最佳实践（本项目踩过的）

1. **`atomicAdd` 无 `signed long long` 重载** → 64 位累加用 `unsigned long long`。
2. **nvcc 版本与 GPU 架构不匹配**（12.4 不识 `sm_120`）→ 用 `compute_90` PTX + 驱动 JIT。
3. **首次计时虚高**（升频 + 首次分配）→ 必须 warmup 并多次取均值。
4. **可分离滤波 in-place 冲突** → 严格遵守 `dst != B` 的乒乓路由。
5. **忘记边界判断** → 三维/一维核都要 `if (越界) return`。
6. **忘记同步就读结果/计时** → 核启动是异步的，需 `cudaDeviceSynchronize`。
7. **未检查错误** → 每个 CUDA 调用用 `CUDA_CHECK` 包裹，并打印 `cudaGetErrorString`。
8. **算法层陷阱**：一维开运算去不掉“沿该轴”的毛刺，去细尖突需三维盒状结构元
   （`rx=ry=rz≥1`）——由 `--selftest` 用例固化。
9. **`int` 溢出**：大 `N` 的线性索引用 `size_t`。

---

## 12. 一个核的完整“解剖”示例

以形态学的 1D min/max 核为例，串起前面所有知识点：
```cuda
__global__ void morphAxisKernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                int W, int H, int D, int axis, int r, bool dilate) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;   // §1.3 线程定位
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    if (x>=W || y>=H || z>=D) return;              // §1.5 边界判断

    uint8_t acc = dilate ? 0u : 1u;                // 膨胀初值0(取max)/腐蚀初值1(取min)
    for (int d = -r; d <= r; ++d) {                // §7 沿单轴的窗口
        int nx=x, ny=y, nz=z;
        if (axis==0)      nx = clampi(x+d,0,W-1);  // §2.3 复制填充
        else if (axis==1) ny = clampi(y+d,0,H-1);  // §7.4 一致分支
        else              nz = clampi(z+d,0,D-1);
        uint8_t v = in[(nz*H + ny)*W + nx];        // §2.1 行主序索引
        acc = dilate ? (v>acc?v:acc) : (v<acc?v:acc);
    }
    out[(z*H + y)*W + x] = acc;
}
```
它体现了：三维网格定位、边界判断、`__restrict__`、复制填充、一致分支、行主序索引、以及
“min=腐蚀 / max=膨胀”的统一实现。

---

## 13. 进一步优化路线（针对本项目）

- **共享内存 tile 加速 stencil**：形态学/高斯的多次 1D 扫描存在重叠读取，可把 tile 载入
  共享内存后在片上多次复用，降低全局访存。
- **常数内存(constant memory)放卷积核**：高斯权重对所有线程相同、只读，放
  `__constant__` 可利用其广播缓存（本项目当前放普通 device 数组，已够用）。
- **流(stream)与异步重叠**：用多 stream 把 H2D、计算、D2H 重叠，隐藏传输延迟。
- **协作组做设备端全局同步**：把并查集收敛循环整体留在 GPU，省去每轮主机往返（§6.6）。
- **CUDA Events 精细计时**：对单个核用事件计时，定位真正热点。
- **marching cubes 网格导出**：在 `d_mask` 上生成 STL/PLY，直接服务三维可视化。

---

### 附：延伸阅读关键词
`CUDA C++ Programming Guide`（Programming Model / Memory Hierarchy / Atomics /
Cooperative Groups）、`CUDA Best Practices Guide`（Memory Optimizations / Timing /
Occupancy）、PTX ISA、`cub::DeviceReduce`、`thrust`。
