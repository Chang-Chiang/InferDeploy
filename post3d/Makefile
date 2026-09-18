# CUDA build for the 3D liver-mask post-processing pipeline.
#
# NOTE ON GPU ARCHITECTURE:
#   The dev machine has an RTX 5070 Ti (Blackwell, sm_120) but the installed
#   toolkit is CUDA 12.4, whose nvcc does not know sm_120. We therefore emit
#   compute_90 PTX and let the (much newer) driver JIT-compile it to the
#   native ISA at runtime. This was verified to work on this machine.
#   Override GENCODE=... on the command line to target a different GPU.
NVCC      ?= nvcc
GENCODE   ?= -gencode arch=compute_90,code=compute_90
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE) -lineinfo -Isrc

# Pure C++ (host-only) toolchain for the CPU reference implementation.
CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra

SRC := src/volume.cu src/ccl.cu src/morphology.cu src/gaussian.cu src/selftest.cu src/main.cu
OBJ := $(SRC:.cu=.o)
HDR := $(wildcard src/*.cuh)
BIN := bin/liver3d
UF  := bin/uf_demo

all: $(BIN)

$(BIN): $(OBJ) | bin
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $@

bin:
	@mkdir -p bin

src/%.o: src/%.cu $(HDR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# CPU reference: pure C++ Union-Find + 3D CCL (no CUDA needed).
uf: $(UF)

$(UF): reference/union_find.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f src/*.o $(BIN) $(UF)

.PHONY: all clean uf
