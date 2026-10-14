// ptx_jit_demo.cu —— PTX / SASS / JIT 的最小可复现样例
//
// 配套文档：docs/PTX_JIT_SASS.zh-CN.md
// 里面两个核只有一行有效逻辑，取自 src/ccl.cu（atomicMin 并查集）与
// src/morphology.cu（clamp 复制填充边界处理），目的是让 PTX/SASS 的输出
// 短到可以逐行读。
//
// 编译（三选一对比产物内容，或看文档第 5 节）：
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=compute_120 -c ptx_jit_demo.cu -o only_ptx.o
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120      -c ptx_jit_demo.cu -o only_sass.o
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=[sm_120,compute_120] -c ptx_jit_demo.cu -o both.o
// 检查产物里到底嵌了什么：
//   cuobjdump --list-elf both.o ; cuobjdump --list-ptx both.o
// 看汇编：
//   cuobjdump -ptx  -fun _Z8uf_unionPiPKii only_ptx.o
//   cuobjdump -sass -fun _Z8uf_unionPiPKii only_sass.o
//
// 直接编译成可执行文件跑起来，可当作「驱动能否 JIT 当前 PTX」的冒烟测试：
//   nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 ptx_jit_demo.cu -o demo_ptxonly
//   ./demo_ptxonly     # 工具链比驱动新时会打印 value=0.0 + unsupported toolchain
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// 与 src/morphology.cu 相同的边界处理：复制（clamp）填充
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// 与 src/ccl.cu 相同的并查集 union：用 atomicMin 把根压小，保证迭代收敛
__global__ void uf_union(int* __restrict__ parent, const int* __restrict__ labels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;  // 越界返回：所有维度都要判
    }
    int hi = labels[i];  // 邻居所属的标签
    if (hi < 0) {
        return;  // 背景跳过
    }
    atomicMin(&parent[hi], i);  // 根只减不增 => 不动点迭代必然终止
}

// 单元素写回：若核函数根本没被执行，host 侧读回来的仍是 memset 的 0
__global__ void writeFlag(int* __restrict__ out) {
    if (threadIdx.x == 0) {
        *out = 1;
    }
}

int main() {
    int            dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    // 这就是文档里说的「运行卡 → sm_XX」的来历：major.minor = 12.0 => sm_120
    std::printf("gpu=%s cc=%d.%d (sm_%d%d)\n", prop.name, prop.major, prop.minor, prop.major, prop.minor);

    constexpr int kN          = 8;
    int           hLabels[kN] = {3, -1, 0, 1, 4, -1, 2, 7};
    int           hParent[kN];
    for (int i = 0; i < kN; ++i) {
        hParent[i] = i;
    }

    int* dParent = nullptr;
    int* dLabels = nullptr;
    int* dFlag   = nullptr;
    cudaMalloc(&dParent, kN * sizeof(int));
    cudaMalloc(&dLabels, kN * sizeof(int));
    cudaMalloc(&dFlag, sizeof(int));
    cudaMemcpy(dLabels, hLabels, kN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dParent, hParent, kN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dFlag, 0, sizeof(int));  // 归零：用来判断核函数有没有真的执行

    // 注意：下面三行在 API 层面全部返回 cudaSuccess，即使核函数被拒绝加载
    uf_union<<<1, kN>>>(dParent, dLabels, kN);
    writeFlag<<<1, 32>>>(dFlag);
    cudaError_t syncErr = cudaDeviceSynchronize();

    int hFlag = 0;
    cudaMemcpy(&hFlag, dFlag, sizeof(int), cudaMemcpyDeviceToHost);  // 这一步必定成功

    const char* lastErr = cudaGetErrorString(cudaGetLastError());
    std::printf("writeFlag executed? value=%d  (期望 1)\n", hFlag);
    std::printf("syncErr=%s\n", cudaGetErrorString(syncErr));
    std::printf("lastErr=%s\n", lastErr);
    std::printf("=> %s\n", (hFlag == 1 && std::string(lastErr) == "no error")
                               ? "核函数已执行"
                               : "核函数未执行（SASS 缺失且 PTX 被拒绝 JIT）");

    cudaFree(dParent);
    cudaFree(dLabels);
    cudaFree(dFlag);
    return (hFlag == 1) ? 0 : 2;
}
