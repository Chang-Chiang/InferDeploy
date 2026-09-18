# 3D-Process — CUDA 3D Organ-Mask Post-Processing Pipeline

Reconstructs **slice-by-slice segmentation masks** (e.g. a 2D UNet run per axial
CT slice) into a **3D body volume** and cleans it with a fully **CUDA**
post-processing pipeline, producing a regular organ mask ready for **volume
statistics** and **3D visualization**.

> Task: *将逐层分割掩膜重建为三维体数据，实现基于 CUDA 的后处理流水线——最大连通域标记与筛选去除层间孤立假阳性区域，3D 形态学开闭弥合层间不连续并消除孔洞毛刺，3D 高斯平滑抑制边缘噪声，输出规整的脏器三维掩膜，支撑后续体积统计与三维可视化。*

Developed and benchmarked on an **NVIDIA RTX 5070 Ti (Blackwell, cc 12.0)**.

> **Detailed algorithm write-up:** [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — the full
> processing flow and every algorithm (parallel union-find CCL, separable box
> morphology, separable Gaussian), with pseudo-code, complexity, correctness
> arguments, timing methodology and benchmarks.

---

## Pipeline

```
 per-slice 2D masks (slice_%04d.u8)
        │
   [1] reconstruct 3D  ── host assembly + H2D upload + binarize  →  uint8 volume (W×H×D)
        │
   [2] 3D CCL + filter ── GPU union-find, 26-connectivity        →  drop components < minSize
        │                                                          (removes inter-layer isolated false positives)
   [3] 3D morphology   ── separable box closing then opening     →  bridge inter-layer gaps, fill holes, remove burrs
        │
   [4] 3D Gaussian     ── separable 3D Gaussian + threshold 0.5  →  suppress edge noise, smooth surface
        │
   [5] finalize        ── D2H, volume stats, write mask + PGM     →  regular organ mask
```

Every stage is timed with a wall-clock stopwatch that synchronizes the device,
and the GPU stages are averaged over `--repeat` runs after a warm-up pass so the
numbers are not skewed by cold-start clock ramp or first-allocation cost.

---

## Directory layout

```
Makefile                  CUDA build (emits compute_90 PTX, driver JITs to sm_120)
run.sh                    build → self-test → generate data → run → validate
src/
  common.cuh              CUDA error checking, grid helper, Stopwatch
  volume.cuh / volume.cu  [1] slice→3D reconstruction, raw/PGM I/O, voxel counting
  ccl.cuh    / ccl.cu     [2] 3D connected-component labeling (union-find) + size filter
  morphology.cuh/.cu      [3] separable 3D box erosion/dilation → open/close
  gaussian.cuh/.cu        [4] separable 3D Gaussian smoothing + threshold
  selftest.cuh/.cu        known-answer unit tests for stages 2–4
  main.cu                 [5] CLI, orchestration, per-step timing report
tools/
  gen_test_data.py        synthetic UNet-style liver masks (numpy only)
  import_masks.py         import real UNet masks (PNG/PGM/PBM/RAW) to pipeline input
tests/
  test_pipeline.py        end-to-end integration test (parses report + numpy checks)
reference/
  union_find.cpp          pure C++ Union-Find + CPU 3D CCL reference (mirrors src/ccl.cu)
docs/
  ALGORITHM.md            detailed algorithm flow + per-algorithm explanation (EN)
  ALGORITHM.zh-CN.md      算法流程与每个算法的详细讲解（中文）
  CUDA_GUIDE.zh-CN.md     以本项目为例的 CUDA 知识详解（中文）
  UNION_FIND.zh-CN.md     并查集详解 + 纯 C++ 实现讲解（中文）
data/                     generated inputs/outputs (git-ignored)
```

---

## Requirements

- **CUDA toolkit** (`nvcc`) 12.x and an NVIDIA GPU.
- **C++17** host compiler, GNU `make`.
- **Python 3** with **numpy** (for test-data generation / import / tests). Pillow is
  optional (only needed to import PNG/JPG/TIFF masks).

### GPU architecture note (Blackwell / sm_120)
This machine has CUDA **12.4**, whose `nvcc` does not yet know `sm_120` (RTX 50
series). The `Makefile` therefore compiles to **`compute_90` PTX** and lets the
newer driver **JIT** it to the native ISA at launch — verified working here.
To target another GPU, override the arch, e.g.:

```bash
make GENCODE="-gencode arch=compute_86,code=sm_86"     # RTX 30 series
make GENCODE="-gencode arch=compute_89,code=sm_89"     # RTX 40 / L40
```

---

## Quick start

```bash
./run.sh                    # build + self-test + generate data + run + validate
./run.sh --minsize 128 --mrz 2 --sigma 1.5 --repeat 10   # args forwarded to the binary
REGEN=1 ./run.sh            # force regeneration of the test data
```

Manual:

```bash
make                                   # -> bin/liver3d
python3 tools/gen_test_data.py         # -> data/slices/*.u8, data/meta.txt, data/liver_gt.u8
./bin/liver3d --selftest               # kernel unit tests
./bin/liver3d                          # run pipeline + print timings
python3 tests/test_pipeline.py         # integration validation
```

---

## Command-line options

| Option                     | Default            | Meaning                                                     |
| -------------------------- | ------------------ | ----------------------------------------------------------- |
| `--slices <dir>`           | `data/slices`      | per-slice raw masks `slice_%04d.u8` (W·H bytes each)        |
| `--meta <file>`            | `data/meta.txt`    | `W H D dx dy dz` (dims + voxel spacing in mm)               |
| `--stack <file>`           | –                  | single contiguous raw stack (overrides `--slices`)          |
| `--gt <file>`              | `data/liver_gt.u8` | ground-truth stack for Dice reporting                       |
| `--no-gt`                  | –                  | skip ground-truth/Dice evaluation                           |
| `--out <prefix>`           | `data/out/liver`   | output prefix (`_final.u8`, `_preview_z*.pgm`)              |
| `--bin <int>`              | `0`                | binarize threshold (foreground where `raw > bin`)           |
| `--minsize <int>`          | `64`               | CCL: remove components smaller than this (voxels)           |
| `--keep-largest`           | off                | CCL: keep only the single largest component                 |
| `--mrx/--mry/--mrz <int>`  | `1 1 1`            | morphology box radii (voxels; raise `mrz` to bridge layers) |
| `--no-close` / `--no-open` | off                | disable morphological closing / opening                     |
| `--sigma <float>`          | `1.0`              | Gaussian sigma (voxels)                                     |
| `--gthr <float>`           | `0.5`              | Gaussian re-binarization threshold                          |
| `--previews <int>`         | `3`                | number of axial PGM previews                                |
| `--repeat <int>`           | `5`                | timing repeats per GPU stage (averaged)                     |
| `--selftest`               | –                  | run CUDA kernel self-tests and exit                         |

---

## Test data

### Synthetic (default, reproducible)
`tools/gen_test_data.py` builds a liver-like ground-truth blob (two ellipsoid
lobes + cardiac notch, ≈1776 cm³ at 1×1×2.5 mm) and a **predicted** mask that
mimics typical slice-wise UNet imperfections:

- wobbly / jittered boundaries (per-slice inference noise)
- **inter-layer discontinuities** (shifted slices + dropout slices → thin gaps)
- interior holes (vessels mislabeled as background)
- **edge burrs** (thin surface protrusions)
- **isolated false positives** (small blobs present in only 1–2 slices)

Outputs `data/slices/slice_*.u8` (the pipeline input), `data/meta.txt`,
`data/liver_pred_stack.u8`, `data/liver_gt.u8`, and `data/preview/*.pgm`.

### Real UNet liver results
Run your UNet on a public liver CT dataset and import its per-slice predictions:

```bash
python3 tools/import_masks.py --indir unet_pred_png --out data \
    --ext '*.png' --dx 0.7 --dy 0.7 --dz 2.0 --threshold 127
./bin/liver3d --no-gt      # or keep --gt if you also import a ground truth
```

`import_masks.py` accepts **PNG/JPG/TIFF** (via Pillow), **PGM/PBM** (pure numpy),
or **RAW** bytes. Useful public sources of liver CT / segmentation masks:

- **3D-IRCADb-01** — 20 abdominal CTs with liver (and tumor) masks. <https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/>
- **LiTS** (Liver Tumor Segmentation Challenge) — 130+ contrast-enhanced CTs. <https://competitions.codalab.org/competitions/17094>

---

## Algorithms

**[2] 3D connected-component labeling (`src/ccl.cu`).** Parallel **union-find**
over 26-connectivity. Each voxel starts as its own root; a *merge* kernel unions
every foreground voxel with its 13 forward neighbours using `atomicMin` on the
root (roots only decrease ⇒ guaranteed termination), and a *compress* kernel does
pointer-jumping. The two alternate until no parent changes (typically 2–4
iterations). A per-root size histogram (`atomicAdd`) then drives the filter:
components with fewer than `minSize` voxels are cleared — this removes the
inter-layer isolated false positives. Component count, largest size and
before/after foreground totals are reported.

**[3] 3D morphology (`src/morphology.cu`).** Binary erosion/dilation with a
**box** structuring element, implemented as **three separable 1D min/max passes**
(X, Y, Z) — O(r) per voxel instead of O(r³). *Closing* (dilate→erode) bridges
inter-layer gaps and fills holes; *opening* (erode→dilate) removes burrs and thin
spikes. Per-axis radii let you elongate the SE along Z to specifically mend
slice-to-slice discontinuities. Replicate (clamp) padding at volume borders.

**[4] 3D Gaussian smoothing (`src/gaussian.cu`).** The mask is converted to
float and convolved with a **separable** 1D Gaussian along X, Y, Z
(kernel radius `⌈3σ⌉`, weights normalized), then re-thresholded at `--gthr`
(default 0.5). This suppresses high-frequency edge noise and yields a smooth,
regular surface while approximately preserving volume.

---

## Benchmark (RTX 5070 Ti, 256×256×128 = 8.39 M voxels, GPU stages averaged over 5 runs)

```
 1a. reconstruct 3D (load slices)           ~1.17 ms     (host file I/O)
 1b. reconstruct 3D (H2D + binarize)        ~0.85 ms
 2.  3D CCL + component filter              ~1.26 ms     (174 → 1 components)
 3.  3D morphology (close + open)           ~1.00 ms
 4.  3D Gaussian smooth + threshold         ~0.66 ms
 5.  finalize (D2H + stats + write)         ~9.6 ms      (dominated by 8 MB file write)
 --------------------------------------------------------------
 TOTAL                                      ~14.6 ms
```

Effect on quality (synthetic UNet-like prediction vs. ground truth):

```
 Dice  reconstructed 0.9845  →  final 0.9876   (+0.0031)
 Organ volume ≈ 1777.7 cm³ (1.78 L)
```

The pure GPU compute (steps 1b–4) is only a few milliseconds for an 8.4 M-voxel
volume; step 5 is I/O bound (writing the mask + previews to disk).

---

## Outputs

- `data/out/liver_final.u8` — the cleaned binary mask (W·H·D bytes, values 0/1),
  directly loadable in **3D Slicer / ITK-SNAP / ParaView** (as raw uint8 with the
  dimensions/spacing from `meta.txt`) for 3D visualization and further analysis.
- `data/out/liver_preview_z###.pgm` — axial PGM previews for a quick look.
- Console report — per-step timings, component statistics, foreground voxel
  counts, **organ volume** (mm³ / cm³ / L) and Dice vs. ground truth.

---

## Validation

- `./bin/liver3d --selftest` — known-answer unit tests: CCL keeps the large block
  and drops a small speck; closing bridges a 1-slice Z gap; opening removes a thin
  spike; Gaussian keeps a solid block and erases an isolated voxel.
- `python3 tests/test_pipeline.py` — runs the binary, then checks that all steps
  reported timings, CCL reduced the component count, the output is strictly
  binary with matching voxel count/volume/Dice, and accuracy did not degrade.
