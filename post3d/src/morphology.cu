/*
 * 可分离的三维二值形态学（box 结构元素）。
 */
#include "common.cuh"
#include "morphology.cuh"

namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// 沿单一轴的一维 min（腐蚀）/ max（膨胀）滤波，边界用复制填充。
// axis：0=X、1=Y、2=Z。dilate：true => max，false => min。
__global__ void morphAxisKernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H, int D,
                                int axis, int r, bool dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= W || y >= H || z >= D) {
        return;
    }

    uint8_t acc = dilate ? 0u : 1u;
    for (int d = -r; d <= r; ++d) {
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
        uint8_t v = in[(nz * H + ny) * W + nx];
        acc       = dilate ? (v > acc ? v : acc) : (v < acc ? v : acc);
    }
    out[(z * H + y) * W + x] = acc;
}

// 依次执行 X -> A、Y -> B、Z -> dst。要求 dst != B（dst 可以等于 src 或 A）。
void separable(const uint8_t* src, uint8_t* A, uint8_t* B, uint8_t* dst, bool dilate, int W, int H, int D, int rx,
               int ry, int rz) {
    dim3 block(8, 8, 8);
    dim3 grid = gridFor(W, H, D, block);
    morphAxisKernel<<<grid, block>>>(src, A, W, H, D, 0, rx, dilate);
    morphAxisKernel<<<grid, block>>>(A, B, W, H, D, 1, ry, dilate);
    morphAxisKernel<<<grid, block>>>(B, dst, W, H, D, 2, rz, dilate);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void morphOpenClose(uint8_t* d_mask, int W, int H, int D, int rx, int ry, int rz, bool doClose, bool doOpen) {
    const size_t N  = (size_t)W * H * D;
    uint8_t*     t1 = nullptr;
    uint8_t*     t2 = nullptr;
    CUDA_CHECK(cudaMalloc(&t1, N));
    CUDA_CHECK(cudaMalloc(&t2, N));

    if (doClose) {                                                  // 先膨胀后腐蚀：弥合层间空隙，填补孔洞
        separable(d_mask, t1, t2, t1, true, W, H, D, rx, ry, rz);   // -> t1
        separable(t1, t2, t1, d_mask, false, W, H, D, rx, ry, rz);  // -> d_mask
    }
    if (doOpen) {                                                   // 先腐蚀后膨胀：去除毛刺 / 细弱突出
        separable(d_mask, t1, t2, t1, false, W, H, D, rx, ry, rz);  // -> t1
        separable(t1, t2, t1, d_mask, true, W, H, D, rx, ry, rz);   // -> d_mask
    }

    CUDA_CHECK(cudaFree(t1));
    CUDA_CHECK(cudaFree(t2));
}
