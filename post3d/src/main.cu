/*
 * 三维肝脏掩码后处理流水线（CUDA）。
 *
 *   逐切面掩码  ->  三维重建
 *               ->  三维连通域标记 + 尺寸过滤
 *               ->  三维形态学闭运算 + 开运算
 *               ->  三维高斯平滑 + 重新阈值化
 *               ->  规整后的器官掩码（+ 体积统计、预览图）
 *
 * 每个阶段都用墙钟秒表计时，读数前会同步设备，因此上报的数字
 * 反映的是真实的端到端单步开销。
 */
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ccl.cuh"
#include "common.cuh"
#include "gaussian.cuh"
#include "morphology.cuh"
#include "selftest.cuh"
#include "volume.cuh"

namespace {

struct Config {
    std::string slicesDir = "data/slices";
    std::string metaFile  = "data/meta.txt";
    std::string stackFile;                          // 可选的单文件输入（优先于 slicesDir）
    std::string gtFile       = "data/liver_gt.u8";  // 可选的金标准，用于算 Dice
    std::string outPrefix    = "data/out/liver";
    int         binThreshold = 0;           // 原始值 > binThreshold 即前景
    int         minSize      = 64;          // CCL：丢弃小于该尺寸的连通域
    bool        keepLargest  = false;       // CCL：只保留最大的一个连通域
    int         mrx = 1, mry = 1, mrz = 1;  // 形态学 box 半径（体素）
    bool        doClose = true, doOpen = true;
    float       sigma      = 1.0f;  // 高斯 sigma（体素）
    float       gthreshold = 0.5f;  // 重新二值化阈值
    int         previews   = 3;     // 写入多少个轴位 PGM 预览图
    int         repeat     = 5;     // 每个 GPU 阶段的计时重复次数（取均值）
    bool        useGt      = true;
    bool        selftest   = false;  // 跑完核函数自检即退出
};

void printUsage() {
    printf("Usage: liver3d [options]\n"
           "  --slices <dir>     per-slice raw masks slice_%%04d.u8 (default data/slices)\n"
           "  --meta <file>      meta text 'W H D dx dy dz'          (default data/meta.txt)\n"
           "  --stack <file>     single contiguous raw stack (overrides --slices)\n"
           "  --gt <file>        ground-truth stack for Dice          (default data/liver_gt.u8)\n"
           "  --no-gt            skip Dice/ground-truth evaluation\n"
           "  --out <prefix>     output prefix                        (default data/out/liver)\n"
           "  --bin <int>        binarize threshold                   (default 0)\n"
           "  --minsize <int>    min connected-component size         (default 64)\n"
           "  --keep-largest     keep only the largest component\n"
           "  --mrx/--mry/--mrz <int> morphology radii               (default 1 1 1)\n"
           "  --no-close         disable morphological closing\n"
           "  --no-open          disable morphological opening\n"
           "  --sigma <float>    Gaussian sigma in voxels             (default 1.0)\n"
           "  --gthr <float>     Gaussian re-binarize threshold       (default 0.5)\n"
           "  --previews <int>   number of axial PGM previews         (default 3)\n"
           "  --repeat <int>     timing repeats per GPU stage         (default 5)\n"
           "  --selftest         run CUDA kernel self-tests and exit\n"
           "  -h | --help        show this help\n");
}

bool parseInt(int argc, char** argv, int& i, int& out) {
    if (i + 1 >= argc) {
        return false;
    }
    out = atoi(argv[++i]);
    return true;
}

bool parseFloat(int argc, char** argv, int& i, float& out) {
    if (i + 1 >= argc) {
        return false;
    }
    out = (float)atof(argv[++i]);
    return true;
}

Config parseArgs(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            printUsage();
            exit(0);
        }
        else if (a == "--slices") {
            c.slicesDir = argv[++i];
        }
        else if (a == "--meta") {
            c.metaFile = argv[++i];
        }
        else if (a == "--stack") {
            c.stackFile = argv[++i];
        }
        else if (a == "--gt") {
            c.gtFile = argv[++i];
        }
        else if (a == "--no-gt") {
            c.useGt = false;
        }
        else if (a == "--out") {
            c.outPrefix = argv[++i];
        }
        else if (a == "--bin") {
            parseInt(argc, argv, i, c.binThreshold);
        }
        else if (a == "--minsize") {
            parseInt(argc, argv, i, c.minSize);
        }
        else if (a == "--keep-largest") {
            c.keepLargest = true;
        }
        else if (a == "--mrx") {
            parseInt(argc, argv, i, c.mrx);
        }
        else if (a == "--mry") {
            parseInt(argc, argv, i, c.mry);
        }
        else if (a == "--mrz") {
            parseInt(argc, argv, i, c.mrz);
        }
        else if (a == "--no-close") {
            c.doClose = false;
        }
        else if (a == "--no-open") {
            c.doOpen = false;
        }
        else if (a == "--sigma") {
            parseFloat(argc, argv, i, c.sigma);
        }
        else if (a == "--gthr") {
            parseFloat(argc, argv, i, c.gthreshold);
        }
        else if (a == "--previews") {
            parseInt(argc, argv, i, c.previews);
        }
        else if (a == "--repeat") {
            parseInt(argc, argv, i, c.repeat);
        }
        else if (a == "--selftest") {
            c.selftest = true;
        }
        else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            printUsage();
            exit(1);
        }
    }
    return c;
}

double diceScore(const uint8_t* a, const uint8_t* b, size_t n) {
    long inter = 0, sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i]) {
            ++sa;
            if (b[i]) {
                ++inter;
            }
        }
        if (b[i]) {
            ++sb;
        }
    }
    if (sa + sb == 0) {
        return 1.0;
    }
    return 2.0 * (double)inter / (double)(sa + sb);
}

struct Step {
    const char* name;
    double      ms;
};

}  // namespace

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);
    if (cfg.selftest) {
        return runSelfTest();
    }
    std::vector<Step> steps;

    // --- CUDA 上下文 / 设备信息（在计时之前完成） -------------------------
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // --- 读取元数据 -------------------------------------------------------
    VolumeMeta meta;
    if (!loadMeta(cfg.metaFile.c_str(), meta)) {
        fprintf(stderr, "failed to read meta: %s\n", cfg.metaFile.c_str());
        return 1;
    }
    const size_t N = meta.voxels();

    // =====================================================================
    // 步骤 1：由逐切面掩码重建三维体
    // =====================================================================
    std::vector<uint8_t> h_raw(N);
    Stopwatch            sw;

    sw.start();
    bool ok;
    if (!cfg.stackFile.empty()) {
        ok = loadStack(cfg.stackFile.c_str(), N, h_raw.data());
    }
    else {
        ok = loadSlicesDir(cfg.slicesDir.c_str(), meta, h_raw.data());
    }
    double tLoad = sw.stopMs();  // 含设备同步，但此时还没有任何 GPU 工作
    if (!ok) {
        fprintf(stderr, "failed to load input masks\n");
        return 1;
    }

    // 预热：先不带计时地跑完整链路一遍，把 GPU 频率拉起来并让缓存
    // 分配器预先就位，这样上报的耗时才具有代表性。
    {
        uint8_t* w = reconstructOnDevice(h_raw.data(), meta, cfg.binThreshold);
        filterComponents(w, meta.W, meta.H, meta.D, cfg.minSize, cfg.keepLargest);
        morphOpenClose(w, meta.W, meta.H, meta.D, cfg.mrx, cfg.mry, cfg.mrz, cfg.doClose, cfg.doOpen);
        gaussianSmoothMask(w, meta.W, meta.H, meta.D, cfg.sigma, cfg.gthreshold);
        CUDA_CHECK(cudaFree(w));
    }

    const int R = cfg.repeat > 0 ? cfg.repeat : 1;

    // 步骤 1b：重建（H2D + 二值化），R 次取均值
    uint8_t* d_mask  = nullptr;
    double   reconMs = 0.0;
    for (int r = 0; r < R; ++r) {
        sw.start();
        uint8_t* m = reconstructOnDevice(h_raw.data(), meta, cfg.binThreshold);
        reconMs += sw.stopMs();
        if (d_mask) {
            CUDA_CHECK(cudaFree(d_mask));
        }
        d_mask = m;
    }
    reconMs /= R;
    steps.push_back({"1a. reconstruct 3D (load slices)", tLoad});
    steps.push_back({"1b. reconstruct 3D (H2D+binarize)", reconMs});

    long reconFg = countForegroundDevice(d_mask, N);

    // 可选的金标准（仅用于上报 Dice，不属于流水线的一环）。
    std::vector<uint8_t> h_gt;
    bool                 haveGt = false;
    if (cfg.useGt) {
        h_gt.resize(N);
        if (loadStack(cfg.gtFile.c_str(), N, h_gt.data())) {
            haveGt = true;
        }
    }
    double diceRecon = haveGt ? diceScore(h_raw.data(), h_gt.data(), N) : -1.0;

    // 未经后处理的重建掩码副本，用于在每次重复之间恢复体数据。
    uint8_t* d_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recon, N));
    CUDA_CHECK(cudaMemcpy(d_recon, d_mask, N, cudaMemcpyDeviceToDevice));

    // 步骤 2-4：每个阶段计时 R 次取均值（每轮开始前先重置体数据）。
    double    cclMs = 0.0, morphMs = 0.0, gaussMs = 0.0;
    CclResult ccl;
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaMemcpy(d_mask, d_recon, N, cudaMemcpyDeviceToDevice));

        sw.start();  // 步骤 2：三维连通域标记 + 剔除孤立假阳性
        ccl = filterComponents(d_mask, meta.W, meta.H, meta.D, cfg.minSize, cfg.keepLargest);
        cclMs += sw.stopMs();

        sw.start();  // 步骤 3：三维形态学闭运算 + 开运算
        morphOpenClose(d_mask, meta.W, meta.H, meta.D, cfg.mrx, cfg.mry, cfg.mrz, cfg.doClose, cfg.doOpen);
        morphMs += sw.stopMs();

        sw.start();  // 步骤 4：三维高斯平滑 + 重新阈值化
        gaussianSmoothMask(d_mask, meta.W, meta.H, meta.D, cfg.sigma, cfg.gthreshold);
        gaussMs += sw.stopMs();
    }
    cclMs /= R;
    morphMs /= R;
    gaussMs /= R;
    steps.push_back({"2.  3D CCL + component filter", cclMs});
    steps.push_back({"3.  3D morphology (close+open)", morphMs});
    steps.push_back({"4.  3D Gaussian smooth + threshold", gaussMs});
    CUDA_CHECK(cudaFree(d_recon));

    // =====================================================================
    // 步骤 5：收尾 -> 取回 host，统计体积，写出结果与预览图
    // =====================================================================
    sw.start();
    std::vector<uint8_t> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_mask, N, cudaMemcpyDeviceToHost));
    long   finalFg = countForegroundDevice(d_mask, N);
    double vmm3    = (double)finalFg * meta.voxelVolumeMm3();

    std::string finalPath = cfg.outPrefix + "_final.u8";
    saveStack(finalPath.c_str(), h_out.data(), N);

    // 轴位 PGM 预览图（等间隔取切面）
    const size_t sliceBytes = (size_t)meta.W * meta.H;
    int          np         = cfg.previews > 0 ? cfg.previews : 0;
    for (int p = 0; p < np; ++p) {
        int  z = (np == 1) ? meta.D / 2 : (int)((long)p * (meta.D - 1) / (np - 1));
        char path[1024];
        snprintf(path, sizeof(path), "%s_preview_z%03d.pgm", cfg.outPrefix.c_str(), z);
        savePGM(path, h_out.data() + (size_t)z * sliceBytes, meta.W, meta.H);
    }
    double tFinalize = sw.stopMs();
    steps.push_back({"5. finalize (D2H+stats+write)", tFinalize});

    double diceFinal = haveGt ? diceScore(h_out.data(), h_gt.data(), N) : -1.0;

    // --- 输出报告 ---------------------------------------------------------
    printf("=============================================================\n");
    printf(" CUDA 3D Liver-Mask Post-Processing Pipeline\n");
    printf("=============================================================\n");
    printf(" GPU            : %s (%d SMs, cc %d.%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf(" Volume         : %d x %d x %d  (%.2f M voxels)\n", meta.W, meta.H, meta.D, N / 1e6);
    printf(" Voxel spacing  : %.3f x %.3f x %.3f mm\n", meta.dx, meta.dy, meta.dz);
    printf("-------------------------------------------------------------\n");
    printf(" Per-step timing (GPU stages averaged over %d runs):\n", R);
    double total = 0.0;
    for (const auto& s : steps) {
        printf("   %-38s %9.3f ms\n", s.name, s.ms);
        total += s.ms;
    }
    printf("   %-38s %9.3f ms\n", "TOTAL pipeline", total);
    printf("-------------------------------------------------------------\n");
    printf(" CCL: components found=%d kept=%d largest=%d voxels (iters=%d)\n", ccl.numComponentsTotal,
           ccl.numComponentsKept, ccl.largestComponent, ccl.iterations);
    printf("      foreground  before filter=%ld  after filter=%ld\n", ccl.foregroundBefore, ccl.foregroundAfter);
    printf(" Foreground voxels: reconstructed=%ld  final=%ld\n", reconFg, finalFg);
    printf(" Organ volume     : %.1f mm^3 = %.3f cm^3 = %.4f L\n", vmm3, vmm3 / 1000.0, vmm3 / 1e6);
    if (haveGt) {
        printf(" Dice vs GT       : reconstructed=%.4f  final=%.4f  (delta %+.4f)\n", diceRecon, diceFinal,
               diceFinal - diceRecon);
    }
    printf(" Output           : %s\n", finalPath.c_str());
    printf("=============================================================\n");

    CUDA_CHECK(cudaFree(d_mask));
    return 0;
}
