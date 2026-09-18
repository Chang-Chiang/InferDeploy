// 3D liver-mask post-processing pipeline (CUDA).
//
//   slice-by-slice masks  ->  3D reconstruction
//                         ->  3D connected-component labeling + size filter
//                         ->  3D morphological closing + opening
//                         ->  3D Gaussian smoothing + threshold
//                         ->  regularized organ mask (+ volume stats, previews)
//
// Each stage is timed with a wall-clock stopwatch that synchronizes the device,
// so the reported numbers reflect real end-to-end per-step cost.
#include "common.cuh"
#include "volume.cuh"
#include "ccl.cuh"
#include "morphology.cuh"
#include "gaussian.cuh"
#include "selftest.cuh"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {

struct Config {
    std::string slicesDir = "data/slices";
    std::string metaFile  = "data/meta.txt";
    std::string stackFile;              // optional single-file input (overrides slicesDir)
    std::string gtFile    = "data/liver_gt.u8";   // optional ground truth for Dice
    std::string outPrefix = "data/out/liver";
    int   binThreshold = 0;             // foreground where raw > binThreshold
    int   minSize      = 64;            // CCL: drop components smaller than this
    bool  keepLargest  = false;         // CCL: keep only the single largest component
    int   mrx = 1, mry = 1, mrz = 1;    // morphology box radii (voxels)
    bool  doClose = true, doOpen = true;
    float sigma = 1.0f;                 // Gaussian sigma (voxels)
    float gthreshold = 0.5f;            // re-binarization threshold
    int   previews = 3;                 // number of axial PGM previews to write
    int   repeat = 5;                   // timing repeats per GPU stage (averaged)
    bool  useGt = true;
    bool  selftest = false;             // run kernel self-tests and exit
};

void printUsage() {
    printf(
      "Usage: liver3d [options]\n"
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
    if (i + 1 >= argc) return false; out = atoi(argv[++i]); return true;
}
bool parseFloat(int argc, char** argv, int& i, float& out) {
    if (i + 1 >= argc) return false; out = (float)atof(argv[++i]); return true;
}

Config parseArgs(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { printUsage(); exit(0); }
        else if (a == "--slices")  c.slicesDir = argv[++i];
        else if (a == "--meta")    c.metaFile  = argv[++i];
        else if (a == "--stack")   c.stackFile = argv[++i];
        else if (a == "--gt")      c.gtFile    = argv[++i];
        else if (a == "--no-gt")   c.useGt     = false;
        else if (a == "--out")     c.outPrefix = argv[++i];
        else if (a == "--bin")     parseInt(argc, argv, i, c.binThreshold);
        else if (a == "--minsize") parseInt(argc, argv, i, c.minSize);
        else if (a == "--keep-largest") c.keepLargest = true;
        else if (a == "--mrx")     parseInt(argc, argv, i, c.mrx);
        else if (a == "--mry")     parseInt(argc, argv, i, c.mry);
        else if (a == "--mrz")     parseInt(argc, argv, i, c.mrz);
        else if (a == "--no-close") c.doClose = false;
        else if (a == "--no-open")  c.doOpen  = false;
        else if (a == "--sigma")   parseFloat(argc, argv, i, c.sigma);
        else if (a == "--gthr")    parseFloat(argc, argv, i, c.gthreshold);
        else if (a == "--previews") parseInt(argc, argv, i, c.previews);
        else if (a == "--repeat")  parseInt(argc, argv, i, c.repeat);
        else if (a == "--selftest") c.selftest = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); printUsage(); exit(1); }
    }
    return c;
}

double diceScore(const uint8_t* a, const uint8_t* b, size_t n) {
    long inter = 0, sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i]) { ++sa; if (b[i]) ++inter; }
        if (b[i]) ++sb;
    }
    if (sa + sb == 0) return 1.0;
    return 2.0 * (double)inter / (double)(sa + sb);
}

struct Step { const char* name; double ms; };

} // namespace

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);
    if (cfg.selftest) return runSelfTest();
    std::vector<Step> steps;

    // --- CUDA context / device banner (done before timing) -------------------
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // --- Load metadata -------------------------------------------------------
    VolumeMeta meta;
    if (!loadMeta(cfg.metaFile.c_str(), meta)) {
        fprintf(stderr, "failed to read meta: %s\n", cfg.metaFile.c_str());
        return 1;
    }
    const size_t N = meta.voxels();

    // =====================================================================
    // STEP 1: reconstruct 3D body from slice-by-slice masks
    // =====================================================================
    std::vector<uint8_t> h_raw(N);
    Stopwatch sw;

    sw.start();
    bool ok;
    if (!cfg.stackFile.empty()) ok = loadStack(cfg.stackFile.c_str(), N, h_raw.data());
    else                        ok = loadSlicesDir(cfg.slicesDir.c_str(), meta, h_raw.data());
    double tLoad = sw.stopMs();     // includes device sync but there is no GPU work yet
    if (!ok) { fprintf(stderr, "failed to load input masks\n"); return 1; }

    // Warmup: run the full chain once (untimed) to ramp GPU clocks and prime
    // the caching allocator, so reported timings are representative.
    {
        uint8_t* w = reconstructOnDevice(h_raw.data(), meta, cfg.binThreshold);
        filterComponents(w, meta.W, meta.H, meta.D, cfg.minSize, cfg.keepLargest);
        morphOpenClose(w, meta.W, meta.H, meta.D, cfg.mrx, cfg.mry, cfg.mrz, cfg.doClose, cfg.doOpen);
        gaussianSmoothMask(w, meta.W, meta.H, meta.D, cfg.sigma, cfg.gthreshold);
        CUDA_CHECK(cudaFree(w));
    }

    const int R = cfg.repeat > 0 ? cfg.repeat : 1;

    // STEP 1b: reconstruction (H2D + binarize), averaged over R runs
    uint8_t* d_mask = nullptr;
    double reconMs = 0.0;
    for (int r = 0; r < R; ++r) {
        sw.start();
        uint8_t* m = reconstructOnDevice(h_raw.data(), meta, cfg.binThreshold);
        reconMs += sw.stopMs();
        if (d_mask) CUDA_CHECK(cudaFree(d_mask));
        d_mask = m;
    }
    reconMs /= R;
    steps.push_back({"1a. reconstruct 3D (load slices)", tLoad});
    steps.push_back({"1b. reconstruct 3D (H2D+binarize)", reconMs});

    long reconFg = countForegroundDevice(d_mask, N);

    // Optional ground truth (for Dice reporting only; not part of the pipeline).
    std::vector<uint8_t> h_gt;
    bool haveGt = false;
    if (cfg.useGt) {
        h_gt.resize(N);
        if (loadStack(cfg.gtFile.c_str(), N, h_gt.data())) haveGt = true;
    }
    double diceRecon = haveGt ? diceScore(h_raw.data(), h_gt.data(), N) : -1.0;

    // Pristine reconstructed mask, used to reset the volume between repeats.
    uint8_t* d_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recon, N));
    CUDA_CHECK(cudaMemcpy(d_recon, d_mask, N, cudaMemcpyDeviceToDevice));

    // STEPS 2-4: each stage timed and averaged over R runs (reset every run).
    double cclMs = 0.0, morphMs = 0.0, gaussMs = 0.0;
    CclResult ccl;
    for (int r = 0; r < R; ++r) {
        CUDA_CHECK(cudaMemcpy(d_mask, d_recon, N, cudaMemcpyDeviceToDevice));

        sw.start();   // STEP 2: 3D CCL + isolated-false-positive filtering
        ccl = filterComponents(d_mask, meta.W, meta.H, meta.D, cfg.minSize, cfg.keepLargest);
        cclMs += sw.stopMs();

        sw.start();   // STEP 3: 3D morphological closing + opening
        morphOpenClose(d_mask, meta.W, meta.H, meta.D, cfg.mrx, cfg.mry, cfg.mrz, cfg.doClose, cfg.doOpen);
        morphMs += sw.stopMs();

        sw.start();   // STEP 4: 3D Gaussian smoothing + threshold
        gaussianSmoothMask(d_mask, meta.W, meta.H, meta.D, cfg.sigma, cfg.gthreshold);
        gaussMs += sw.stopMs();
    }
    cclMs /= R; morphMs /= R; gaussMs /= R;
    steps.push_back({"2.  3D CCL + component filter", cclMs});
    steps.push_back({"3.  3D morphology (close+open)", morphMs});
    steps.push_back({"4.  3D Gaussian smooth + threshold", gaussMs});
    CUDA_CHECK(cudaFree(d_recon));

    // =====================================================================
    // STEP 5: finalize -> download, volume stats, write outputs & previews
    // =====================================================================
    sw.start();
    std::vector<uint8_t> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_mask, N, cudaMemcpyDeviceToHost));
    long finalFg = countForegroundDevice(d_mask, N);
    double vmm3 = (double)finalFg * meta.voxelVolumeMm3();

    std::string finalPath = cfg.outPrefix + "_final.u8";
    saveStack(finalPath.c_str(), h_out.data(), N);

    // axial PGM previews (evenly spaced slices)
    const size_t sliceBytes = (size_t)meta.W * meta.H;
    int np = cfg.previews > 0 ? cfg.previews : 0;
    for (int p = 0; p < np; ++p) {
        int z = (np == 1) ? meta.D / 2 : (int)((long)p * (meta.D - 1) / (np - 1));
        char path[1024];
        snprintf(path, sizeof(path), "%s_preview_z%03d.pgm", cfg.outPrefix.c_str(), z);
        savePGM(path, h_out.data() + (size_t)z * sliceBytes, meta.W, meta.H);
    }
    double tFinalize = sw.stopMs();
    steps.push_back({"5. finalize (D2H+stats+write)", tFinalize});

    double diceFinal = haveGt ? diceScore(h_out.data(), h_gt.data(), N) : -1.0;

    // --- Report ------------------------------------------------------------
    printf("=============================================================\n");
    printf(" CUDA 3D Liver-Mask Post-Processing Pipeline\n");
    printf("=============================================================\n");
    printf(" GPU            : %s (%d SMs, cc %d.%d)\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);
    printf(" Volume         : %d x %d x %d  (%.2f M voxels)\n", meta.W, meta.H, meta.D, N / 1e6);
    printf(" Voxel spacing  : %.3f x %.3f x %.3f mm\n", meta.dx, meta.dy, meta.dz);
    printf("-------------------------------------------------------------\n");
    printf(" Per-step timing (GPU stages averaged over %d runs):\n", R);
    double total = 0.0;
    for (const auto& s : steps) { printf("   %-38s %9.3f ms\n", s.name, s.ms); total += s.ms; }
    printf("   %-38s %9.3f ms\n", "TOTAL pipeline", total);
    printf("-------------------------------------------------------------\n");
    printf(" CCL: components found=%d kept=%d largest=%d voxels (iters=%d)\n",
           ccl.numComponentsTotal, ccl.numComponentsKept, ccl.largestComponent, ccl.iterations);
    printf("      foreground  before filter=%ld  after filter=%ld\n",
           ccl.foregroundBefore, ccl.foregroundAfter);
    printf(" Foreground voxels: reconstructed=%ld  final=%ld\n", reconFg, finalFg);
    printf(" Organ volume     : %.1f mm^3 = %.3f cm^3 = %.4f L\n",
           vmm3, vmm3 / 1000.0, vmm3 / 1e6);
    if (haveGt) {
        printf(" Dice vs GT       : reconstructed=%.4f  final=%.4f  (delta %+.4f)\n",
               diceRecon, diceFinal, diceFinal - diceRecon);
    }
    printf(" Output           : %s\n", finalPath.c_str());
    printf("=============================================================\n");

    CUDA_CHECK(cudaFree(d_mask));
    return 0;
}
