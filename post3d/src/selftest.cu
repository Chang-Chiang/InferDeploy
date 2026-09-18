// Known-answer unit tests for the three CUDA stages (CCL, morphology, Gaussian).
// Each test builds a tiny volume with a hand-computed expected result, runs the
// stage on the GPU, and verifies the outcome.
#include "selftest.cuh"
#include "common.cuh"
#include "ccl.cuh"
#include "morphology.cuh"
#include "gaussian.cuh"

#include <cstdio>
#include <vector>
#include <cstdint>

namespace {

int g_fail = 0;

void check(bool cond, const char* msg) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) ++g_fail;
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

inline int idx(int x, int y, int z, int W, int H) { return (z * H + y) * W + x; }

// ---- Test 1: CCL keeps the big block, drops the small isolated speck -------
void testCcl() {
    printf("[selftest] 3D connected-component labeling + size filter\n");
    const int W = 8, H = 8, D = 8;
    std::vector<uint8_t> v(W * H * D, 0);
    // 4x4x4 solid block (64 voxels) at the low corner
    for (int z = 0; z < 4; ++z) for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x)
        v[idx(x, y, z, W, H)] = 1;
    // isolated 2-voxel speck far from the block
    v[idx(6, 6, 6, W, H)] = 1;
    v[idx(7, 6, 6, W, H)] = 1;

    uint8_t* d = toDev(v);
    CclResult r = filterComponents(d, W, H, D, /*minSize=*/10, /*keepLargestOnly=*/false);
    auto out = fromDev(d, v.size());
    CUDA_CHECK(cudaFree(d));

    long fg = 0; for (auto b : out) fg += b;
    check(r.numComponentsTotal == 2, "found exactly 2 components");
    check(r.largestComponent == 64, "largest component == 64 voxels");
    check(r.numComponentsKept == 1, "kept exactly 1 component (minSize=10)");
    check(fg == 64, "small speck removed, block retained (fg==64)");
    check(out[idx(6, 6, 6, W, H)] == 0 && out[idx(0, 0, 0, W, H)] == 1,
          "speck voxel cleared, block voxel kept");
}

// ---- Test 2: closing bridges a 1-slice Z gap; opening removes a burr -------
void testMorphology() {
    printf("[selftest] 3D morphology (closing bridges gap, opening removes burr)\n");
    const int W = 8, H = 8, D = 8;

    // (a) closing along Z: two slabs separated by an empty slice z=3
    {
        std::vector<uint8_t> v(W * H * D, 0);
        for (int z = 1; z <= 2; ++z) for (int y = 2; y <= 5; ++y) for (int x = 2; x <= 5; ++x)
            v[idx(x, y, z, W, H)] = 1;
        for (int z = 4; z <= 5; ++z) for (int y = 2; y <= 5; ++y) for (int x = 2; x <= 5; ++x)
            v[idx(x, y, z, W, H)] = 1;
        check(v[idx(3, 3, 3, W, H)] == 0, "gap voxel empty before closing");
        uint8_t* d = toDev(v);
        morphOpenClose(d, W, H, D, /*rx=*/0, /*ry=*/0, /*rz=*/1, /*doClose=*/true, /*doOpen=*/false);
        auto out = fromDev(d, v.size());
        CUDA_CHECK(cudaFree(d));
        check(out[idx(3, 3, 3, W, H)] == 1, "closing bridged the inter-layer gap");
    }

    // (b) 3D opening: a solid block with a thin 1-voxel spike (burr) on its face.
    //     A full 3x3x3 box SE erodes the thin spike away and dilation cannot
    //     regenerate it, so opening removes it while keeping the block.
    {
        std::vector<uint8_t> v(W * H * D, 0);
        for (int z = 2; z <= 5; ++z) for (int y = 2; y <= 5; ++y) for (int x = 2; x <= 5; ++x)
            v[idx(x, y, z, W, H)] = 1;   // 4x4x4 block
        v[idx(6, 3, 3, W, H)] = 1;       // thin spike burr protruding in +x
        uint8_t* d = toDev(v);
        morphOpenClose(d, W, H, D, /*rx=*/1, /*ry=*/1, /*rz=*/1, /*doClose=*/false, /*doOpen=*/true);
        auto out = fromDev(d, v.size());
        CUDA_CHECK(cudaFree(d));
        check(out[idx(6, 3, 3, W, H)] == 0, "opening removed the thin burr/spike");
        check(out[idx(3, 3, 3, W, H)] == 1, "opening kept the solid block interior");
        check(out[idx(5, 3, 3, W, H)] == 1, "opening preserved the block face");
    }
}

// ---- Test 3: Gaussian smoothing keeps a solid block, erases an isolated dot --
void testGaussian() {
    printf("[selftest] 3D Gaussian smoothing + threshold\n");
    const int W = 12, H = 12, D = 12;
    std::vector<uint8_t> v(W * H * D, 0);
    for (int z = 2; z <= 9; ++z) for (int y = 2; y <= 9; ++y) for (int x = 2; x <= 9; ++x)
        v[idx(x, y, z, W, H)] = 1;   // 8^3 solid block
    v[idx(0, 0, 0, W, H)] = 1;       // isolated single voxel (edge noise)

    uint8_t* d = toDev(v);
    gaussianSmoothMask(d, W, H, D, /*sigma=*/1.0f, /*threshold=*/0.5f);
    auto out = fromDev(d, v.size());
    CUDA_CHECK(cudaFree(d));

    check(out[idx(5, 5, 5, W, H)] == 1, "block interior survives smoothing");
    check(out[idx(0, 0, 0, W, H)] == 0, "isolated voxel suppressed (edge noise removed)");
}

} // namespace

int runSelfTest() {
    CUDA_CHECK(cudaFree(0));
    printf("=============================================================\n");
    printf(" CUDA kernel self-test\n");
    printf("=============================================================\n");
    testCcl();
    testMorphology();
    testGaussian();
    printf("-------------------------------------------------------------\n");
    if (g_fail == 0) printf(" SELFTEST: ALL PASSED\n");
    else             printf(" SELFTEST: %d CHECK(S) FAILED\n", g_fail);
    printf("=============================================================\n");
    return g_fail;
}
