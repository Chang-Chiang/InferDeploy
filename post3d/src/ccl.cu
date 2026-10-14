/*
 * 基于并行并查集的三维连通域标记（26 邻接），随后做基于尺寸的过滤，
 * 以剔除跨层的孤立假阳性。
 *
 * 算法：
 *   1. 对每个体素初始化 parent[i] = i。
 *   2. 反复迭代直到稳定：
 *        merge    ：每个前景体素与它的 13 个「前向」前景邻居做 union
 *                  （即可覆盖 26 邻接的全部边一次）。union 在根上做
 *                  atomicMin，根只会越变越小，由此保证终止。
 *        compress：把每个 parent 指针跳转（路径压缩）到其根。
 *   3. label(i) = root(i)；用 atomicAdd 构造按根统计的尺寸直方图。
 *   4. 保留尺寸 >= minSize 的连通域（或只保留最大的那一个）。
 */
#include <cstdio>

#include "ccl.cuh"
#include "common.cuh"

namespace {

__device__ __forceinline__ int uf_find(const int* parent, int x) {
    int r = x;
    while (parent[r] != r) {
        r = parent[r];
    }
    return r;
}

__global__ void initParentKernel(int* parent, size_t N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        parent[i] = (int)i;
    }
}

// 把每个前景体素与它的 13 个前向邻居做 union（26 邻接）。
__global__ void mergeKernel(const uint8_t* __restrict__ mask, int* parent, int W, int H, int D, int* changed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }
    int i = (z * H + y) * W + x;
    if (!mask[i]) {
        return;
    }

    // 26 邻域的 13 个前向偏移（避免同一条边被处理两次）：
    //   前 7 个指向同一层（dz = 0）内的前向邻居，后 6 个指向下一层（dz = 1）
    const int off[13][3] = {
        {1,  0,  0},
        {1,  1,  0},
        {1,  -1, 0},
        {0,  1,  0},
        {0,  -1, 0},
        {-1, 1,  0},
        {-1, -1, 0},
        {1,  0,  1},
        {1,  1,  1},
        {1,  -1, 1},
        {0,  1,  1},
        {0,  -1, 1},
        {0,  0,  1}
    };
    int ri = uf_find(parent, i);
#pragma unroll
    for (int k = 0; k < 13; ++k) {
        int nx = x + off[k][0], ny = y + off[k][1], nz = z + off[k][2];
        if (nx < 0 || ny < 0 || nz < 0 || nx >= W || ny >= H || nz >= D) {
            continue;
        }
        int j = (nz * H + ny) * W + nx;
        if (!mask[j]) {
            continue;
        }
        int rj = uf_find(parent, j);
        if (ri != rj) {
            int lo = min(ri, rj), hi = max(ri, rj);
            atomicMin(&parent[hi], lo);
            ri       = lo;
            *changed = 1;
        }
    }
}

__global__ void compressKernel(int* parent, size_t N, int* changed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    int r = uf_find(parent, (int)i);
    if (parent[i] != r) {
        parent[i] = r;
        *changed  = 1;
    }
}

__global__ void buildCountsKernel(const uint8_t* __restrict__ mask, const int* parent, int* counts, int W, int H,
                                  int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }
    int i = (z * H + y) * W + x;
    if (!mask[i]) {
        return;
    }
    atomicAdd(&counts[uf_find(parent, i)], 1);
}

struct DevStats {
    int                numComp;
    int                keptBySize;
    unsigned long long keptFg;
    unsigned long long totalFg;
    unsigned long long packedMax;  // (count << 32) | label
};

__global__ void statsKernel(const int* counts, size_t N, int minSize, DevStats* s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    int c = counts[i];
    if (c <= 0) {
        return;
    }
    atomicAdd(&s->numComp, 1);
    atomicAdd(&s->totalFg, (unsigned long long)c);
    if (c >= minSize) {
        atomicAdd(&s->keptBySize, 1);
        atomicAdd(&s->keptFg, (unsigned long long)c);
    }
    unsigned long long key = ((unsigned long long)(unsigned)c << 32) | (unsigned)i;
    atomicMax(&s->packedMax, key);
}

__global__ void filterKernel(uint8_t* mask, const int* parent, const int* counts, int W, int H, int D, int minSize,
                             int largestLabel, bool keepLargestOnly) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }
    int i = (z * H + y) * W + x;
    if (!mask[i]) {
        return;
    }
    int  lab  = uf_find(parent, i);
    bool keep = keepLargestOnly ? (lab == largestLabel) : (counts[lab] >= minSize);
    if (!keep) {
        mask[i] = 0;
    }
}

}  // namespace

CclResult filterComponents(uint8_t* d_mask, int W, int H, int D, int minSize, bool keepLargestOnly) {
    const size_t N = (size_t)W * H * D;
    CclResult    res;

    int*      parent    = nullptr;
    int*      counts    = nullptr;
    int*      d_changed = nullptr;
    DevStats* d_stats   = nullptr;
    CUDA_CHECK(cudaMalloc(&parent, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&counts, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(DevStats)));
    CUDA_CHECK(cudaMemset(counts, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(DevStats)));

    dim3     block(8, 8, 8);
    dim3     grid     = gridFor(W, H, D, block);
    int      threads  = 256;
    unsigned blocks1d = (unsigned)((N + threads - 1) / threads);

    initParentKernel<<<blocks1d, threads>>>(parent, N);

    // 迭代执行 merge + compress，直到没有任何 parent 发生变化。
    int iter = 0;
    while (true) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice));
        mergeKernel<<<grid, block>>>(d_mask, parent, W, H, D, d_changed);
        compressKernel<<<blocks1d, threads>>>(parent, N, d_changed);
        int ch = 0;
        CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        ++iter;
        if (!ch) {
            break;
        }
        if (iter > 500) {
            fprintf(stderr, "[ccl] WARN: not converged after %d iters\n", iter);
            break;
        }
    }
    res.iterations = iter;

    buildCountsKernel<<<grid, block>>>(d_mask, parent, counts, W, H, D);
    statsKernel<<<blocks1d, threads>>>(counts, N, minSize, d_stats);

    DevStats hs;
    CUDA_CHECK(cudaMemcpy(&hs, d_stats, sizeof(DevStats), cudaMemcpyDeviceToHost));

    int largestLabel = (int)(hs.packedMax & 0xffffffffull);
    int largestCount = (int)(hs.packedMax >> 32);

    filterKernel<<<grid, block>>>(d_mask, parent, counts, W, H, D, minSize, largestLabel, keepLargestOnly);

    res.numComponentsTotal = hs.numComp;
    res.largestComponent   = largestCount;
    res.foregroundBefore   = (long)hs.totalFg;
    res.foregroundAfter    = keepLargestOnly ? (long)largestCount : (long)hs.keptFg;
    res.numComponentsKept  = keepLargestOnly ? (hs.numComp > 0 ? 1 : 0) : hs.keptBySize;

    CUDA_CHECK(cudaFree(parent));
    CUDA_CHECK(cudaFree(counts));
    CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_stats));
    return res;
}
