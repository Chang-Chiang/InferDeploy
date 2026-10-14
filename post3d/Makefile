# 三维肝脏掩码后处理流水线的 CUDA 构建脚本。
#
# 关于 GPU 架构的说明：
#   开发机是 RTX 5070 Ti（Blackwell，sm_120 / cc 12.0），配 CUDA 13.4 + 驱动 595.99.02。
#   nvcc 13.4 已经认识 sm_120，所以我们直接生成原生 SASS，完全不走驱动 JIT。
#
#   历史沿革：在 CUDA 12.4（其 nvcc 不认识 sm_120）下，我们曾生成 compute_90 的
#   PTX 再交给驱动 JIT。这条兜底路径在当前机器上已失效：CUDA 13.x 输出的
#   PTX ISA 为 9.4，而驱动只接受到 CUDA 13.2，因此纯 PTX 构建会在启动核函数时报
#   "the provided PTX was compiled with an unsupported toolchain"。
#   经验法则：工具链比驱动新时，必须内嵌 SASS。
#
#   要面向别的 GPU，在命令行覆盖 GENCODE=... 即可。
NVCC      ?= nvcc
GENCODE   ?= -gencode arch=compute_120,code=sm_120
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) -lineinfo -Isrc

# cuobjdump 优先用 PATH 里的，否则取与 nvcc 同目录的
CUOBJDUMP ?= $(shell command -v cuobjdump 2>/dev/null || echo $(dir $(shell command -v $(NVCC) 2>/dev/null))cuobjdump)

# 与仓库根 Makefile 的 check_sass 同一约定：产物必须嵌入与 GENCODE 匹配的原生 SASS。
# 两层检查：① GENCODE 里根本没有 sm_XX（纯 PTX 目标）本身就是错；② 有则必须真出现在产物里。
# 纯 PTX 目标在“工具链比驱动新”时会让核函数静默不执行（只有 cudaGetLastError 有内容）。
# 做纯 PTX 对照实验时（docs/PTX_JIT_SASS.zh-CN.md）用 SKIP_SASS_CHECK=1 绕过。
SASS_ARCH := $(shell echo "$(GENCODE)" | grep -oE "sm_[0-9]+" | head -1)
define check_sass
	@if [ -n "$(SKIP_SASS_CHECK)" ]; then exit 0; fi; \
	if [ -z "$(SASS_ARCH)" ]; then \
		echo "ERROR: 构建目标未带任何 sm_XX 原生 SASS（纯 PTX 目标），核函数在工具链比驱动新时会静默不执行；对照实验请用 SKIP_SASS_CHECK=1"; exit 1; \
	fi; \
	if ! $(CUOBJDUMP) --list-elf $(1) 2>/dev/null | grep -q "$(SASS_ARCH)"; then \
		echo "ERROR: $(1) 未嵌入 $(SASS_ARCH) 原生 SASS，核函数会静默不执行（检查 GENCODE）"; exit 1; \
	fi
endef

# 用于 CPU 参考实现的纯 C++（仅 host）工具链。
CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra

SRC := src/volume.cu src/ccl.cu src/morphology.cu src/gaussian.cu src/selftest.cu src/main.cu
OBJ := $(SRC:.cu=.o)
HDR := $(wildcard src/*.cuh)
BIN := bin/liver3d
UF  := bin/uf_demo
DEMO := bin/ptx_jit_demo

all: $(BIN)

$(BIN): $(OBJ) | bin
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $@

bin:
	@mkdir -p bin

# 主流水线的目标码必须带原生 SASS；ptxdemo 故意不加守卫（它就是用来做纯 PTX 对照实验的）
src/%.o: src/%.cu $(HDR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	$(call check_sass,$@)

# CPU 参考实现：纯 C++ 并查集 + 三维连通域标记（不需要 CUDA）。
uf: $(UF)

$(UF): reference/union_find.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

# PTX/SASS/JIT 最小样例（docs/PTX_JIT_SASS.zh-CN.md）：直接跑起来可验证
# 「当前驱动能否 JIT 当前工具链生成的 PTX」；换 GENCODE 对比两种结果。
ptxdemo: $(DEMO)

$(DEMO): reference/ptx_jit_demo.cu | bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f src/*.o $(BIN) $(UF) $(DEMO)

.PHONY: all clean uf ptxdemo
