/*
 * 可分离的三维高斯平滑 + 重新阈值化。
 */
#include <cmath>
#include <cstdio>
#include <vector>

#include "common.cuh"
#include "gaussian.cuh"

namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void toFloatKernel(const uint8_t* __restrict__ mask, float* __restrict__ f, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        f[i] = mask[i] ? 1.0f : 0.0f;
    }
}

__global__ void gaussAxisKernel(const float* __restrict__ in, float* __restrict__ out, int W, int H, int D, int axis,
                                const float* __restrict__ kern, int kr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }

    float acc = 0.f;
    for (int d = -kr; d <= kr; ++d) {
        int nx = x, ny = y, nz = z;
        if (axis == 0) {
            nx = clampi(x + d, 0, W - 1);
        }
        else if (axis == 1) {
            ny = clampi(y + d, 0, H - 1);
        }
        else {
            nz = clampi(z + d, 0, D - 1);
        }
        acc += in[(nz * H + ny) * W + nx] * kern[d + kr];
    }
    out[(z * H + y) * W + x] = acc;
}

__global__ void thresholdKernel(const float* __restrict__ f, uint8_t* __restrict__ mask, size_t n, float thr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        mask[i] = (f[i] >= thr) ? 1u : 0u;
    }
}

}  // namespace

void gaussianSmoothMask(uint8_t* d_mask, int W, int H, int D, float sigma, float threshold) {
    const size_t N = (size_t)W * H * D;
    if (sigma <= 0.f) {
        return;  // 无需处理
    }

    int kr = (int)std::ceil(3.0f * sigma);
    if (kr < 1) {
        kr = 1;
    }
    int klen = 2 * kr + 1;

    // 在 host 侧构造归一化的一维高斯核，只上传一次。
    std::vector<float> h(klen);
    double             sum = 0.0;
    for (int d = -kr; d <= kr; ++d) {
        double w  = std::exp(-(double)(d * d) / (2.0 * sigma * sigma));
        h[d + kr] = (float)w;
        sum += w;
    }
    for (int i = 0; i < klen; ++i) {
        h[i] = (float)(h[i] / sum);
    }

    float* d_kern = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kern, klen * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kern, h.data(), klen * sizeof(float), cudaMemcpyHostToDevice));

    float* f1 = nullptr;
    float* f2 = nullptr;
    CUDA_CHECK(cudaMalloc(&f1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&f2, N * sizeof(float)));

    int      threads  = 256;
    unsigned blocks1d = (unsigned)((N + threads - 1) / threads);
    dim3     block(8, 8, 8);
    dim3     grid = gridFor(W, H, D, block);

    toFloatKernel<<<blocks1d, threads>>>(d_mask, f1, N);
    gaussAxisKernel<<<grid, block>>>(f1, f2, W, H, D, 0, d_kern, kr);  // X
    gaussAxisKernel<<<grid, block>>>(f2, f1, W, H, D, 1, d_kern, kr);  // Y
    gaussAxisKernel<<<grid, block>>>(f1, f2, W, H, D, 2, d_kern, kr);  // Z
    thresholdKernel<<<blocks1d, threads>>>(f2, d_mask, N, threshold);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_kern));
    CUDA_CHECK(cudaFree(f1));
    CUDA_CHECK(cudaFree(f2));
}
