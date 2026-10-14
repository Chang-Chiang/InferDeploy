/*
 * 针对三个 CUDA 步骤（连通域标记、形态学、高斯）的已知答案单元测试。
 * 每个测试构造一个小规模的体数据，其期望结果由人工算出，然后在 GPU 上
 * 跑对应步骤并校验输出。
 */
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ccl.cuh"
#include "common.cuh"
#include "gaussian.cuh"
#include "morphology.cuh"
#include "selftest.cuh"

namespace {

int g_fail = 0;

void check(bool cond, const char* msg) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) {
        ++g_fail;
    }
}

uint8_t* toDev(const std::vector<uint8_t>& h) {
    uint8_t* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size()));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    return d;
}

std::vector<uint8_t> fromDev(uint8_t* d, size_t n) {
    std::vector<uint8_t> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost));
    return h;
}

inline int idx(int x, int y, int z, int W, int H) {
    return (z * H + y) * W + x;
}

// ---- 测试 1：CCL 保留大块连通域，剔除小的孤立碎块 -----------------
void testCcl() {
    printf("[selftest] 3D connected-component labeling + size filter\n");
    const int            W = 8, H = 8, D = 8;
    std::vector<uint8_t> v(W * H * D, 0);
    // 左下角一个 4x4x4 实心块（64 个体素）
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                v[idx(x, y, z, W, H)] = 1;
            }
        }
    }
    // 远离该块的一个孤立 2 体素碎块
    v[idx(6, 6, 6, W, H)] = 1;
    v[idx(7, 6, 6, W, H)] = 1;

    uint8_t*  d   = toDev(v);
    CclResult r   = filterComponents(d, W, H, D, /*minSize=*/10, /*keepLargestOnly=*/false);
    auto      out = fromDev(d, v.size());
    CUDA_CHECK(cudaFree(d));

    long fg = 0;
    for (auto b : out) {
        fg += b;
    }
    check(r.numComponentsTotal == 2, "found exactly 2 components");
    check(r.largestComponent == 64, "largest component == 64 voxels");
    check(r.numComponentsKept == 1, "kept exactly 1 component (minSize=10)");
    check(fg == 64, "small speck removed, block retained (fg==64)");
    check(out[idx(6, 6, 6, W, H)] == 0 && out[idx(0, 0, 0, W, H)] == 1, "speck voxel cleared, block voxel kept");
}

// ---- 测试 2：闭运算弥合 1 个切面的 Z 向间隙；开运算去除毛刺 -------
void testMorphology() {
    printf("[selftest] 3D morphology (closing bridges gap, opening removes burr)\n");
    const int W = 8, H = 8, D = 8;

    // (a) 沿 Z 做闭运算：两个板状区域被一个空白切面 z=3 隔开
    {
        std::vector<uint8_t> v(W * H * D, 0);
        for (int z = 1; z <= 2; ++z) {
            for (int y = 2; y <= 5; ++y) {
                for (int x = 2; x <= 5; ++x) {
                    v[idx(x, y, z, W, H)] = 1;
                }
            }
        }
        for (int z = 4; z <= 5; ++z) {
            for (int y = 2; y <= 5; ++y) {
                for (int x = 2; x <= 5; ++x) {
                    v[idx(x, y, z, W, H)] = 1;
                }
            }
        }
        check(v[idx(3, 3, 3, W, H)] == 0, "gap voxel empty before closing");
        uint8_t* d = toDev(v);
        morphOpenClose(d, W, H, D, /*rx=*/0, /*ry=*/0, /*rz=*/1,
                       /*doClose=*/true, /*doOpen=*/false);
        auto out = fromDev(d, v.size());
        CUDA_CHECK(cudaFree(d));
        check(out[idx(3, 3, 3, W, H)] == 1, "closing bridged the inter-layer gap");
    }

    // (b) 3D 开运算：实心块的某个表面上带着一根 1 体素的细刺（毛刺）。
    //     完整的 3x3x3 box 结构元素会把细刺腐蚀掉，而后续膨胀也无法
    //     使其重生，因此开运算能去刺同时保留整个块。
    {
        std::vector<uint8_t> v(W * H * D, 0);
        for (int z = 2; z <= 5; ++z) {
            for (int y = 2; y <= 5; ++y) {
                for (int x = 2; x <= 5; ++x) {
                    v[idx(x, y, z, W, H)] = 1;  // 4x4x4 实心块
                }
            }
        }
        v[idx(6, 3, 3, W, H)] = 1;  // 沿 +x 方向伸出的 1 体素细刺毛刺
        uint8_t* d            = toDev(v);
        morphOpenClose(d, W, H, D, /*rx=*/1, /*ry=*/1, /*rz=*/1,
                       /*doClose=*/false, /*doOpen=*/true);
        auto out = fromDev(d, v.size());
        CUDA_CHECK(cudaFree(d));
        check(out[idx(6, 3, 3, W, H)] == 0, "opening removed the thin burr/spike");
        check(out[idx(3, 3, 3, W, H)] == 1, "opening kept the solid block interior");
        check(out[idx(5, 3, 3, W, H)] == 1, "opening preserved the block face");
    }
}

// ---- 测试 3：高斯平滑保留实心块，抹掉孤立单点 ----------------------
void testGaussian() {
    printf("[selftest] 3D Gaussian smoothing + threshold\n");
    const int            W = 12, H = 12, D = 12;
    std::vector<uint8_t> v(W * H * D, 0);
    for (int z = 2; z <= 9; ++z) {
        for (int y = 2; y <= 9; ++y) {
            for (int x = 2; x <= 9; ++x) {
                v[idx(x, y, z, W, H)] = 1;  // 8^3 实心块
            }
        }
    }
    v[idx(0, 0, 0, W, H)] = 1;  // 孤立单个体素（边缘噪声）

    uint8_t* d = toDev(v);
    gaussianSmoothMask(d, W, H, D, /*sigma=*/1.0f, /*threshold=*/0.5f);
    auto out = fromDev(d, v.size());
    CUDA_CHECK(cudaFree(d));

    check(out[idx(5, 5, 5, W, H)] == 1, "block interior survives smoothing");
    check(out[idx(0, 0, 0, W, H)] == 0, "isolated voxel suppressed (edge noise removed)");
}

}  // namespace

int runSelfTest() {
    CUDA_CHECK(cudaFree(0));
    printf("=============================================================\n");
    printf(" CUDA kernel self-test\n");
    printf("=============================================================\n");
    testCcl();
    testMorphology();
    testGaussian();
    printf("-------------------------------------------------------------\n");
    if (g_fail == 0) {
        printf(" SELFTEST: ALL PASSED\n");
    }
    else {
        printf(" SELFTEST: %d CHECK(S) FAILED\n", g_fail);
    }
    printf("=============================================================\n");
    return g_fail;
}
