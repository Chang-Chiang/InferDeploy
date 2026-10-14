# PTX / SASS / JIT 详解 —— 以 CUDA 13.4 + RTX 5070 Ti 实测为例

本文解释 CUDA 编译与执行的三层表示（CUDA C++ → **PTX** → **SASS**）、运行期的
**JIT** 到底发生在哪一步、`-gencode arch=…,code=…` 的每个字段含义，以及本项目
实际踩过的坑：**工具链比驱动新时，「嵌 PTX 让驱动 JIT」这条兜底路会彻底失效**。
所有输出都是本机重跑得到的，配套最小样例见
[`../reference/ptx_jit_demo.cu`](../reference/ptx_jit_demo.cu)（`make ptxdemo`）。

> 相关文档：体系结构编译在 [`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md) 第 10 节有简版；
> 本文是展开版。算法本身见 [`ALGORITHM.zh-CN.md`](ALGORITHM.zh-CN.md)，
> 并查集见 [`UNION_FIND.zh-CN.md`](UNION_FIND.zh-CN.md)。

---

## 0. 实测环境（本文所有输出均出自此环境）

| 项目                    | 值                                                              | 查询方式                                                   |
| ----------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| GPU                     | NVIDIA GeForce RTX 5070 Ti（Blackwell）                         | `nvidia-smi --query-gpu=name --format=csv,noheader`        |
| 计算能力 / 真实架构     | **cc 12.0 → `sm_120`**                                          | `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` |
| 工具链                  | **CUDA 13.4**（`nvcc V13.4.92`，`/usr/local/cuda → cuda-13.4`） | `nvcc --version`                                           |
| 驱动                    | **595.99.02**（`nvidia-smi` 报 **CUDA 上限 13.2**）             | `cat /proc/driver/nvidia/version`                          |
| PTX ISA（由工具链决定） | **9.4**                                                         | `cuobjdump -ptx <obj> \| grep -E '^\.version\|^\.target'`  |
| 采集时间                | 2026-10-13                                                      | —                                                          |

TL;DR（三条命令看懂全部结论）：

```bash
make ptxdemo                      # 默认 -gencode arch=compute_120,code=sm_120（原生 SASS）
./bin/ptx_jit_demo                # => 核函数已执行, exit 0

make -B ptxdemo GENCODE="-gencode arch=compute_90,code=compute_90"   # 纯 PTX + 驱动 JIT
./bin/ptx_jit_demo                # => 核函数未执行：unsupported toolchain, exit 2
```

---

## 1. 全景：三个层次，两个编译器

```
 CUDA C++ 源码           ../src/ccl.cu / ../src/morphology.cu / reference/ptx_jit_demo.cu
      │  ① cicc（nvcc 内部的 device 前端 + LLVM 优化）
      ▼
 PTX  虚拟 ISA，文本      .version 9.4 / .target sm_120  →  atom.global.min.s32 …
      │  ② ptxas（GPU 汇编器：分配物理寄存器、调度、生成真实编码）
      ▼
 SASS 真实 ISA，二进制    sm_120 的 cubin（ELF）        →  IMAD / LDG.E / @P0 EXIT …
      │  ③ fatbinary：把 cubin 和/或 PTX 打成 fatbin，随 host 目标码链接进可执行文件
      ▼
 运行期 libcuda 加载：找到匹配 SASS → 直接执行；否则尝试 JIT 编 PTX；都不行 → 报错
```

| 名词                                 | 是什么                                                                   | 类比（CPU 世界）        |
| ------------------------------------ | ------------------------------------------------------------------------ | ----------------------- |
| **PTX**（Parallel Thread Execution） | NVIDIA 对外承诺稳定的**虚拟指令集**，文本可读，跨 GPU 代际               | LLVM IR / Java bytecode |
| **SASS**（Shader Assembly）          | 某一代 GPU 的**真实机器码**，编码/指令集逐代变化，只由工具链内部文档描述 | x86 ELF 机器码          |
| **cubin**                            | 装着 SASS 的 ELF 容器（`*.sm_120.cubin`）                                | `.o`/`.so`              |
| **fatbin**                           | 一个二进制里同时塞多份 cubin / PTX，运行期挑最匹配的                     | universal binary        |
| **JIT**                              | 驱动在自己进程里把 PTX 现编成当前 GPU 的 SASS，并缓存                    | JVM 的 JIT              |
| **`sm_120` / `compute_120`**         | 真实架构名 / 虚拟架构名（后者决定 PTX 与其可用特性）                     | —                       |

---

## 2. PTX：虚拟指令集长什么样

样例源码（并查集 union，与 `src/ccl.cu` 同款原语）：

```cpp
__global__ void uf_union(int* __restrict__ parent, const int* __restrict__ labels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;  // 越界返回
    }
    int hi = labels[i];  // 邻居标签
    if (hi < 0) {
        return;  // 背景跳过
    }
    atomicMin(&parent[hi], i);  // 根只减不增 => 不动点迭代必然收敛
}
```

`nvcc -gencode arch=compute_120,code=compute_120` 后 `cuobjdump -ptx` 的输出（原文照录）：

```ptx
.visible .entry _Z8uf_unionPiPKii(
.param .u64 .ptr .align 1 _Z8uf_unionPiPKii_param_0,
.param .u64 .ptr .align 1 _Z8uf_unionPiPKii_param_1,
.param .u32 _Z8uf_unionPiPKii_param_2
)
{
.reg .pred %p<3>;
.reg .b32 %r<8>;
.reg .b64 %rd<9>;

ld.param.b64 %rd1, [_Z8uf_unionPiPKii_param_0];
ld.param.b64 %rd2, [_Z8uf_unionPiPKii_param_1];
ld.param.b32 %r3, [_Z8uf_unionPiPKii_param_2];
mov.u32 %r4, %ctaid.x;
mov.u32 %r5, %ntid.x;
mov.u32 %r6, %tid.x;
mad.lo.s32 %r1, %r4, %r5, %r6;
setp.ge.s32 %p1, %r1, %r3;
@%p1 bra $L__BB0_3;
cvta.to.global.u64 %rd3, %rd2;
mul.wide.s32 %rd4, %r1, 4;
add.s64 %rd5, %rd3, %rd4;
ld.global.nc.b32 %r2, [%rd5];
setp.lt.s32 %p2, %r2, 0;
@%p2 bra $L__BB0_3;
cvta.to.global.u64 %rd6, %rd1;
mul.wide.u32 %rd7, %r2, 4;
add.s64 %rd8, %rd6, %rd7;
atom.global.min.s32 %r7, [%rd8], %r1;
$L__BB0_3:
ret;
}
```

怎么读它：

| PTX 片段                          | 含义                                                          | 备注                                          |
| --------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| `.version 9.4` / `.target sm_120` | PTX **ISA 版本** / **至少需要的虚拟架构**                     | 这两行是第 6 节坑的全部根源                   |
| `.reg .b32 %r<8>`                 | 声明 8 个 32 位**虚拟**寄存器                                 | 想声明多少都行，物理分配是 ptxas 的事         |
| `%ctaid.x` / `%ntid.x` / `%tid.x` | `blockIdx.x` / `blockDim.x` / `threadIdx.x`                   | 特殊寄存器                                    |
| `mad.lo.s32`                      | `i = blockIdx.x*blockDim.x + threadIdx.x`                     | PTX 里是「乘加」伪指令，真实编码可能是 `IMAD` |
| `setp.ge.s32 %p1, …` + `@%p1 bra` | 比较置谓词 + **谓词化跳转**                                   | 小分支不烧流水线                              |
| `ld.global.nc.b32`                | 从 **global** 内存只读加载（`nc` = non-coherent，走只读缓存） | 对应 `__restrict__` 的读                      |
| `atom.global.min.s32`             | 全局原子取小 —— 并查集收敛性的硬件保证                        | 见 `UNION_FIND.zh-CN.md`                      |
| `mul.wide.s32` + `add.s64`        | 指针算术（下标 × 4 再加基址）                                 | 32 位下标要升到 64 位，`int` 溢出的坑由此可见 |

要点：**PTX 是「架构下界 + 无限虚拟寄存器 + 明确存储类型」的中间语言**，它不关心
你有多少物理寄存器、 LSU 怎么编码，因此才有跨代兼容的资本。

---

## 3. SASS：真实机器码长什么样

同一份源码用 `code=sm_120` 编译后，`cuobjdump -sass` 的输出（节选）：

```
		Function : _Z8uf_unionPiPKii
	.headerflags	@"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM120)"
/*0000*/   LDC          R1, c[0x0][0x37c] ;
/*0010*/   S2R          R0, SR_TID.X ;
/*0020*/   S2UR         UR4, SR_CTAID.X ;
/*0030*/   LDCU         UR5, c[0x0][0x390] ;
/*0050*/   IMAD         R5, R5, UR4, R0 ;
/*0060*/   ISETP.GE.AND P0, PT, R5, UR5, PT ;
/*0070*/   @P0          EXIT ;
/*00b0*/   LDG.E.CONSTANT R7, desc[UR4][R2.64] ;
/*00c0*/   ISETP.GE.AND P0, PT, R7, RZ, PT ;
/*00d0*/   @!P0         EXIT ;
```

PTX → SASS 的映射关系（能读出的硬件事实）：

| PTX                           | SASS                                 | 说明了什么                                                        |
| ----------------------------- | ------------------------------------ | ----------------------------------------------------------------- |
| `mov.u32 %r6, %tid.x`         | `S2R R0, SR_TID.X`                   | 特殊寄存器有专门指令，一条硬件读                                  |
| `mad.lo.s32`                  | `IMAD`                               | 真实乘加单元；整数乘加是单发射                                    |
| `ld.param.b32 … [_param_2]`   | `LDC` / `LDCU c[0x0][…]`             | **kernel 参数通过常量 bank 传递**（所以参数按常量读最快）         |
| `ld.global.nc.b32`            | `LDG.E.CONSTANT desc[UR4][R2.64]`    | 真实的 LSU 指令 + **memory descriptor**（Blackwell 起的编码方式） |
| `bra $L__BB0_3`               | `@P0 EXIT`                           | ptxas 把「跳到 ret」优化成谓词化提前退出                          |
| `.reg .b32 %r<8>`（随便声明） | `Used 10 registers, used 0 barriers` | 物理寄存器由 ptxas 分配（`-Xptxas -v` 实测，见第 9 节）           |
| 无                            | 每条指令后两列 `/* 0x… */`           | 指令的**真实二进制编码**与调度控制位                              |

为什么 SASS 不能跨架构：`desc[UR4][R2.64]` 这类 descriptor 编码、`UR`（统一寄存器）
体系、指令位宽都是**这一代硬件专属**；换一代 `sm_` 就是一套新编码。所以：

> **`sm_86` 的 cubin 绝对跑不了 `sm_120`，反之亦然。**（规则 A，见第 6 节）

---

## 4. 编译流水线与运行期加载

`nvcc --dryrun` 在本机实际调用的工具（照录工具名）：

```
CUDA C++ ─► cudafe++   分离 host/device、C++ 前端
        ─► cicc        device 端优化，输出 PTX（虚拟架构由 -arch 决定）
        ─► ptxas       PTX → cubin(SASS)，寄存器分配/调度（可 -v 看资源）
        ─► fatbinary   把 cubin/PTX 打包成 fatbin
        ─► gcc         编译 host 部分，fatbin 作为 .nv_fatbin 段一并链接
```

运行期（`libcuda.so`，即驱动）在第一次使用某模块时做决策：

```
需要加载 fatbin
  ├─ 有当前 GPU 的 sm_XX cubin？        → 直接搬上 GPU 执行（无编译，最快）
  ├─ 否则有 PTX 且 ISA ≤ 驱动支持？      → JIT 编成 SASS，结果缓存到 ~/.nv/ComputeCache
  └─ 都不满足                           → 失败：
       cudaErrorNoKernelImageForDevice  ("no kernel image is available for execution on the device")
       cudaErrorUnsupportedToolchain    ("the provided PTX was compiled with an unsupported toolchain.")
```

注意：**核函数是异步启动的**，失败信息不会打断 `cudaMemcpy` 之类的调用，
详见第 7 节（这正是它难被查出的原因）。

---

## 5. `-gencode` 逐字段与三种写法

```
-gencode arch=<虚拟架构>,code=<要嵌入的产物>
           │                   ├─ sm_XX      → 跑 ptxas，嵌 SASS(cubin)
           │                   ├─ compute_XX → 嵌 PTX 文本（可被驱动 JIT）
           │                   └─ [sm_XX,compute_XX] → 两者都嵌
           └─ 决定 PTX 的 .target 与可用特性（如 compute_90 不能用 Blackwell 新指令）
```

实测三种写法产物的**内容**差异（`cuobjdump --list-elf / --list-ptx`）：

| 写法                                         | `--list-elf`               | `--list-ptx`             | 在 5070 Ti 上的行为                       |
| -------------------------------------------- | -------------------------- | ------------------------ | ----------------------------------------- |
| `arch=compute_120,code=compute_120`          | *No ELF file found*        | `demo_ptx.1.sm_120.ptx`  | 只能靠 JIT（驱动接受才可用）              |
| `arch=compute_120,code=sm_120`               | `demo_sass.1.sm_120.cubin` | *No PTX file found*      | 直接执行 ← **`post3d/Makefile` 当前默认** |
| `arch=compute_120,code=[sm_120,compute_120]` | `demo_both.1.sm_120.cubin` | `demo_both.1.sm_120.ptx` | 本机走 cubin，PTX 留给更新的驱动          |

几条容易混淆的规则：

- `nvcc -arch=sm_120` 是简写，等价于 `arch=compute_120,code=sm_120`（**只有 SASS，没有 PTX 兜底**）。
- 一个 fatbin 里可以放**多代** cubin：重复写 `-gencode`（发布通用包的做法，代价是编译时间与产物大小）。
- 后缀变体：`sm_120a`（arch-accelerated，只用本卡独占指令，**互不兼容**）、
  `sm_120f`（family-specific，同家族兼容）。部署只用不带后缀的最稳。
- 本项目换卡：`make GENCODE="-gencode arch=compute_89,code=sm_89"`（`GENCODE ?=` 可覆盖）。
- **别拿体积当选型依据**：实测两种方向都出现过 —— demo 里 PTX 版 `10,024 B` < SASS 版
  `18,336 B`；而 `bin/liver3d` 的 PTX 版是 `2,685,480 B` > SASS 版 `1,371,856 B`。
  取决于文本压缩率、核函数数量与嵌入架构数。

---

## 6. 三条兼容规则，以及本项目踩坑的完整时间线

### 规则 A：SASS **不跨架构**（既不向前也不向后）
要匹配运行卡的真实架构 `sm_XX`。

### 规则 B：PTX 只保证「**向前**兼容」
`compute_90` 生成的 PTX 可以在 `sm_120` 上被 JIT。因为 PTX 头的
`.target sm_90` 是「**至少需要** sm_90 的特性」——**下界**而非上界。

### 规则 C（致命细节）：PTX 的 `.version` 必须 **≤ 驱动支持的 PTX ISA**
`.version` 由**工具链版本**决定，驱动只内置了「≤ 自身 CUDA 上限」的 PTX 编译器：

| 产出方          | PTX `.version` | 本机驱动 595.99.02（上限 CUDA 13.2）能否 JIT |
| --------------- | -------------- | -------------------------------------------- |
| CUDA 12.4（旧） | `8.4`          | ✅ 能（仓库里的旧产物至今仍可运行）           |
| CUDA 13.4（新） | `9.4`          | ❌ 拒绝 → `unsupported toolchain`             |

### 时间线复盘

| 阶段           | 工具链                  | 当时的做法                                                                           | 现在结果                             |
| -------------- | ----------------------- | ------------------------------------------------------------------------------------ | ------------------------------------ |
| ① 2026-09      | CUDA 12.4 + 驱动 595.84 | `nvcc` 不识 `sm_120` → 发 `compute_90` PTX，靠新驱动 JIT（规则 B 生效）              | 旧 `bin/liver3d` 仍可跑              |
| ② 2026-10 升级 | CUDA 13.4 + 驱动 595.99 | 同样写法撞上规则 C：13.4 的 PTX ISA 9.4 > 驱动上限 13.2 → **JIT 被拒**               | `--selftest` 全 FAIL                 |
| ③ 现在         | CUDA 13.4               | 改为原生 SASS `arch=compute_120,code=sm_120`（13.4 认识 `compute_120`，规则 A 满足） | `make`/`--selftest`/`run.sh` 全 PASS |

**结论一句话**：*JIT 兜底只在「工具链 ≤ 驱动」时成立；工具链更新时，必须嵌目标架构的 SASS。*

判断只需查三处：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # 运行卡：12.0 ⇒ sm_120
nvcc --list-gpu-arch                                    # nvcc 认识哪些虚拟架构（13.4 从 compute_75 起步）
nvidia-smi | head -3                                    # 驱动右上角的 CUDA 上限：13.2 ⇒ PTX 只能 ≤ 它
```

> 附带变化：CUDA 13 已**移除** `compute_60/61/70/72` 等旧架构，沿用旧 `-gencode`
> 会在编译期就报 `nvcc fatal : Unsupported gpu architecture 'compute_61'`
> （主工程 `config/Makefile.config` 的 `ARCH` 就踩过这条）。

---

## 7. 最难查的部分：失败是**静默**的

同一份 `reference/ptx_jit_demo.cu`，两种目标，运行输出只差最后一行（实测照录）：

```
# make ptxdemo（原生 SASS）
gpu=NVIDIA GeForce RTX 5070 Ti cc=12.0 (sm_120)
writeFlag executed? value=1  (期望 1)
syncErr=no error
lastErr=no error
=> 核函数已执行
exit=0

# make -B ptxdemo GENCODE="-gencode arch=compute_90,code=compute_90"（纯 PTX）
gpu=NVIDIA GeForce RTX 5070 Ti cc=12.0 (sm_120)
writeFlag executed? value=0  (期望 1)
syncErr=no error
lastErr=the provided PTX was compiled with an unsupported toolchain.
=> 核函数未执行（SASS 缺失且 PTX 被拒绝 JIT）
exit=2
```

逐条拆解这次暴露的语义：

1. `cudaMalloc / cudaMemset / cudaMemcpy` **全部成功**——它们不碰核函数代码。
2. `<<< >>>` 启动失败被记录在 **last-error** 状态里，**launch 本身没有返回值可查**。
3. 本例中 `cudaDeviceSynchronize()` 返回 `no error`，而 `cudaGetLastError()` 才有内容
   ——所以**只看同步返回值会漏掉这类错误**，两者都要查。
4. 于是数据仍是 `memset` 的 0：现象是「程序跑完了、没报错、结果全不对」。

本项目里的真实表现：`--selftest` 里连通域数量算成 0（一连串 `[FAIL]`），
直到 `src/morphology.cu:40` 的 `CUDA_CHECK` 才把错误串打印出来。防御三原则：

| 做法                                          | 本项目落点                                                   |
| --------------------------------------------- | ------------------------------------------------------------ |
| 每次核启动后取 `cudaGetLastError()`（包进宏） | `../src/common.cuh` 的 `CUDA_CHECK`                          |
| **校验结果本身**，而不是只校验 API 返回值     | `../src/selftest.cu` 已知答案单测 + `tests/test_pipeline.py` |
| CI/验收必须**从源码重编**再跑，别只测旧二进制 | `./run.sh`（build → selftest → 生成数据 → 跑 → 校验）        |

---

## 8. JIT 的代价：延迟、缓存与可复现性

- **首次启动延迟**：JIT 发生在第一次 launch，编译时长随核函数规模增长。计时必须 warmup
  的另一个成因（本项目实测记录见 [`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md) 第 9 节）。
- **缓存位置**：`~/.nv/ComputeCache`（本机确实存在）。所以「第二次跑就快了」可能是
  缓存的功劳，而不是你的优化生效 —— 做冷启动对比时要显式关掉。
- **懒加载**：CUDA 12+ 默认 `CUDA_MODULE_LOADING=LAZY`，只在实际用到某核时才编译/上传，
  初始化更快；想一次性暴露所有 JIT 问题就设 `EAGER`。
- **调试信息**：JIT 产物与 `-lineinfo` 的行号映射不如直接生成的 cubin 稳定。

常用环境变量：

| 变量                         | 作用                     | 典型用法                               |
| ---------------------------- | ------------------------ | -------------------------------------- |
| `CUDA_MODULE_LOADING`        | `LAZY`/`EAGER`           | 排障时 `EAGER`，一次性把加载问题全暴露 |
| `CUDA_CACHE_DISABLE=1`       | 禁用 JIT 缓存            | 冷启动计时，避免缓存污染数据           |
| `CUDA_CACHE_MAXSIZE=<bytes>` | 缓存上限（有最小值限制） | 长跑服务防止缓存无限增长               |

想看驱动/JIT 的内部日志时：新版驱动靠 `CUDA_LOG_SEVERITY` / `CUDA_LOG_FILENAME` 一对变量（需
驱动日志组件就位），而常见的 `CUDA_LOG_LEVEL` 在本机实测**不会产生任何输出**——排错优先靠
第 7 节的 `cudaGetLastError()` 与 `-Xptxas -v`，别指望日志变量。

---

## 9. 排障工具箱（本机验证可用）

```bash
# 0) 我是谁：卡 / 驱动 / 工具链
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvidia-smi | head -3                 # CUDA Version（驱动上限）
nvcc --version                       # 工具链版本
nvcc --list-gpu-arch                 # 我能编哪些虚拟架构

# 1) 产物里到底嵌了什么（部署前必查这两条）
cuobjdump --list-elf bin/liver3d     # 有哪些架构的 SASS
cuobjdump --list-ptx bin/liver3d     # 有没有 PTX 兜底

# 2) 读汇编
cuobjdump -ptx  -fun _Z8uf_unionPiPKii only_ptx.o   # 单个核的 PTX
cuobjdump -sass -fun _Z8uf_unionPiPKii only_sass.o  # 单个核的 SASS（实测可用）
# nvdisasm 只接受**裸 cubin**，得先拆出来（直接丢 .o 会报 not a supported Elf file）：
cuobjdump -xelf all only_sass.o                      # → only_sass.1.sm_120.cubin
nvdisasm only_sass.1.sm_120.cubin                    # 带控制流/标签的完整反汇编视图

# 3) 编译期资源与耗时（本项目寄存器压力就靠它看）
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Xptxas -v -c reference/ptx_jit_demo.cu -o /dev/null
#   ptxas info : Compiling entry function '_Z8uf_unionPiPKii' for 'sm_120'
#   ptxas info : Used 10 registers, used 0 barriers     ←  occupancy 估算输入
#   ptxas info : Compile time = 0.941 ms
#   （同法可见 clamp_pad 用 8 registers）

# 4) 看 nvcc 内部流程（不真正执行）
nvcc --dryrun -gencode arch=compute_120,code=[sm_120,compute_120] -c reference/ptx_jit_demo.cu -o x.o
```

---

## 10. 决策表，以及和其他组件的边界

| 我的场景                                  | 该用什么                                                               |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| 只在本机/同代卡跑，要最快启动             | `-gencode arch=compute_120,code=sm_120`（本项目当前默认）              |
| 发布给若干已知卡                          | 逐个加 `-gencode …,code=sm_XX`（SASS 不跨架构）                        |
| 需要兼容未来更新的卡，且**驱动 ≥ 工具链** | 额外嵌 PTX：`code=[sm_XX,compute_XX]`                                  |
| `nvcc` 太老不认新卡                       | **优先升工具链**；升不动才用老 `compute_` PTX 靠新驱动 JIT             |
| 工具链比驱动新（本项目的坑）              | 只能嵌目标架构 SASS；或降工具链 / 升驱动                               |
| 报 `no kernel image is available`         | 没有匹配 cubin **且** 没有可 JIT 的 PTX → 检查 `--list-elf/--list-ptx` |
| 报 `unsupported toolchain`                | PTX `.version` 高于驱动上限 → 见第 6 节规则 C                          |
| 不报错但结果没变                          | 见第 7 节：查 `cudaGetLastError()` + 跑结果级 selftest                 |

与本项目其他部分的边界（很容易混）：

- `config/Makefile.config` 的 `ARCH` **只影响你自己的 `.cu`**（主工程仅 `src/utils/Process.cu`，
  即预处理核函数）。**TensorRT 的 kernel 由 TRT 自己选择/编译**，与这个 `ARCH` 无关。
- `post3d` 是纯 CUDA C++、不涉 TRT，所以它是这套规则里唯一会直接踩「工具链 vs 驱动」错配的地方。
- 动态链接层是**另一个坑**，别和这里混起来：`libnvinfer.so.10` 的 `NEEDED` 里**没有**
  cudart/cublas（实测），所以升级到 CUDA 13.4 后 TRT 库本身可解析；卡住主工程的是
  `CUDA_DIR := /usr` 失去 apt 路径布局（链接报 `cannot find -lcudart/-lcublas`）与
  `libcublas.so.12` 缺失（`ldd` 显示 `not found`）。详见
  [`../../docs/目标检测部署流程.md`](../../docs/目标检测部署流程.md) 的「环境已变更」说明。

---

## 11. 一页速查

```
层次：CUDA C++ ─(cicc)→ PTX(虚拟,文本,.version/.target) ─(ptxas)→ SASS(真实,cubin) ─(fatbinary)→ fatbin
运行：驱动优先找匹配 sm_XX 的 cubin；否则 JIT PTX（结果进 ~/.nv/ComputeCache）；都不行才报错

-gencode arch=compute_XX  → 我要哪一代 PTX 特性（下界）
         code=sm_XX       → 嵌 SASS（快、不跨架构）
         code=compute_XX  → 嵌 PTX（可 JIT，但要求 工具链 ≤ 驱动）

三张表：nvidia-smi compute_cap → 目标 sm_XX
        nvcc --list-gpu-arch   → 工具链能否产出它
        nvidia-smi CUDA 上限   → 驱动能否 JIT 我的 PTX .version

本项目：CUDA 13.4 + 驱动 595.99 ⇒ GENCODE = -gencode arch=compute_120,code=sm_120
```

---

## 附：复现校验

```bash
cd post3d
make ptxdemo && ./bin/ptx_jit_demo; echo $?                     # 0，核函数已执行
make -B ptxdemo GENCODE="-gencode arch=compute_90,code=compute_90"
./bin/ptx_jit_demo; echo $?                                     # 2，unsupported toolchain
make -B ptxdemo                                                 # 恢复默认（原生 SASS）

cuobjdump --list-elf bin/liver3d | head -3                       # 全是 *.sm_120.cubin
cuobjdump --list-ptx bin/liver3d                                 # No PTX file found
./bin/liver3d --selftest                                         # SELFTEST: ALL PASSED
./run.sh                                                         # ALL CHECKS PASSED
```
