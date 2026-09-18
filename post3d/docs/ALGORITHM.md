# Algorithm Documentation — CUDA 3D Organ-Mask Post-Processing Pipeline

This document explains, step by step, the full processing flow and every
algorithm used inside it. It is the companion to the code in [`../src`](../src).
(中文版：[`ALGORITHM.zh-CN.md`](ALGORITHM.zh-CN.md))

> **Task (原始需求).** *将逐层分割掩膜重建为三维体数据，实现基于 CUDA 的后处理流水线——
> 最大连通域标记与筛选去除层间孤立假阳性区域，3D 形态学开闭弥合层间不连续并消除孔洞毛刺，
> 3D 高斯平滑抑制边缘噪声，输出规整的脏器三维掩膜，支撑后续体积统计与三维可视化。*

In English: reconstruct slice-by-slice segmentation masks into a 3D body, then run
a CUDA post-processing pipeline — (1) largest/connected-component labeling and
size filtering to remove inter-layer isolated false positives, (2) 3D
morphological open/close to bridge inter-layer discontinuity and eliminate holes
and burrs, (3) 3D Gaussian smoothing to suppress edge noise — and output a
regular organ mask that supports volume statistics and 3D visualization.

---

## 0. Data representation and notation

| Symbol | Meaning |
|---|---|
| `W, H, D` | volume width (x), height (y), depth / number of slices (z) |
| `N = W·H·D` | total voxel count |
| `dx, dy, dz` | voxel spacing in mm (from `meta.txt`) |
| `M[i] ∈ {0,1}` | binary mask, stored as `uint8`, 1 = foreground (organ) |

**Memory layout (row-major, z-slowest):**

\[ i = (z \cdot H + y)\cdot W + x \]

All 3D kernels map a thread to `(x,y,z)` via a `dim3 block(8,8,8)` /
`gridFor(W,H,D)` grid ([`common.cuh`](../src/common.cuh)); all 1D kernels use
256 threads over `N`. Boundary access uses **replicate (clamp)** padding:
`clampi(v, 0, dim-1)`.

**Pipeline order** ([`main.cu`](../src/main.cu)):

```
slices → [1] reconstruct → [2] CCL + filter → [3] close+open → [4] Gaussian → [5] finalize
```

Each stage operates **in place** on the device mask `d_mask` (except stage 1,
which produces it, and stage 5, which reads it).

---

## Stage 1 — 3D reconstruction from slice masks

**File:** [`volume.cu`](../src/volume.cu) · **Entry:** `reconstructOnDevice()`

### Purpose
Turn the per-slice 2D outputs of a 2D segmentation network (e.g. a UNet run on
each axial CT slice) into a single normalized 3D binary volume.

### Input format
- `data/slices/slice_%04d.u8` — one raw `W·H`-byte file per slice (values `0/1`
  or `0..255`), **or** a single contiguous stack via `--stack`.
- `data/meta.txt` — `W H D dx dy dz`.

### Algorithm
1. **Host assembly** (`loadSlicesDir`): read the `D` slice files sequentially into
   one host buffer `h_raw[z·W·H … ]`. This is the literal "stacking" of layers.
2. **Upload + binarize** (`reconstructOnDevice`):
   - `cudaMalloc` `d_raw`, `d_mask` (N bytes each); `cudaMemcpy` `h_raw → d_raw`.
   - `binarizeKernel`: for each voxel, `d_mask[i] = (d_raw[i] > binThreshold) ? 1 : 0`.
   - free `d_raw`; return `d_mask`.

```
binarizeKernel(raw, out, N, thr):
    i = global_thread_id
    if i < N: out[i] = (raw[i] > thr)
```

### Cost
- Time: `O(N)` memory-bound copy + `O(N)` binarize; dominated by host file I/O and H2D PCIe transfer.
- Memory: transient `2·N` bytes on device.

The thresholding here also normalizes arbitrary mask encodings (e.g. `0/255`
PNG-exported slices) to a clean `{0,1}` volume for the downstream stages.

---

## Stage 2 — 3D connected-component labeling + size filtering

**File:** [`ccl.cu`](../src/ccl.cu) · **Entry:** `filterComponents()`

### Purpose
Label every 3D connected region of the mask, measure each region's size, and
delete the small ones — these are the **inter-layer isolated false positives**
(specks that appear on one or two slices and are not connected to the organ).

### Why union-find on the GPU
Classic sequential CCL (two-pass scan with a merge table) is inherently serial.
On the GPU we use a **parallel union-find**: every voxel is its own set; adjacent
foreground voxels are unioned; path compression flattens the forest. The
canonical label of a component becomes the **minimum voxel index** in it, which
gives a natural, stable id and guarantees termination (labels only ever shrink).

### Connectivity
**26-connectivity** (full 3×3×3 neighbourhood minus centre). To process each
undirected edge exactly once, the merge kernel visits only the **13 forward
offsets**:

```
{+1,0,0} {+1,+1,0} {+1,-1,0} {0,+1,0} {0,-1,0} {-1,+1,0} {-1,-1,0}
{+1,0,+1} {+1,+1,+1} {+1,-1,+1} {0,+1,+1} {0,-1,+1} {0,0,+1}
```

### Data structures
- `parent[N]` (`int`) — union-find forest, `parent[i]` initially `i`.
- `counts[N]` (`int`) — per-root size histogram.
- `DevStats` — a few device scalars reduced with atomics.

### Kernel sequence

**a. `initParentKernel`** — `parent[i] = i` for all `i` (1D over `N`).

**b. `mergeKernel`** (3D over foreground voxels) — union with forward neighbours:

```
i = voxel; if !M[i] return
ri = find(parent, i)
for each forward offset o:
    j = neighbour(i, o); if out-of-bounds or !M[j] continue
    rj = find(parent, j)
    if ri != rj:
        lo = min(ri, rj); hi = max(ri, rj)
        atomicMin(&parent[hi], lo)     # link the larger root to the smaller
        ri = lo
        changed = 1
```

`find` (`uf_find`) is a **read-only** iterative walk to the root — it does not
compress, so concurrent threads never race on writes during the merge.
`atomicMin` makes the union safe under concurrency and monotone: a root value can
only decrease, so the iteration is guaranteed to converge.

**c. `compressKernel`** (1D over `N`) — pointer jumping:

```
r = find(parent, i)
if parent[i] != r: parent[i] = r; changed = 1
```

**d. Convergence loop** (host): reset `changed=0`, run merge, run compress, read
`changed` back; repeat until no parent changed (safety cap 500). Compression
halves path lengths each round, so the number of iterations is small in practice
(**measured: 2–4**, e.g. 3 for the liver volume).

**e. `buildCountsKernel`** (3D) — size histogram:

```
if M[i]: atomicAdd(&counts[ find(parent, i) ], 1)
```

After convergence `find(i)` is the component's minimum index, so all voxels of a
component increment the same slot.

**f. `statsKernel`** (1D over `counts`) — one pass computes:
- `numComp`   — number of components (`counts[i] > 0`);
- `totalFg`   — Σ counts = foreground voxels before filtering;
- `keptBySize`, `keptFg` — components / voxels with `counts[i] ≥ minSize`;
- `packedMax` — the largest component **and** its label in a single atomic:

\[ \text{key} = (\text{count} \ll 32)\;|\;\text{label}, \qquad \texttt{atomicMax}(\text{packedMax}, \text{key}) \]

Because count occupies the high 32 bits, `atomicMax` on `unsigned long long`
selects the biggest component; the low 32 bits then give its label. (Note: CUDA
has **no** `atomicAdd(long long*)` overload — the accumulators are
`unsigned long long`.)

**g. `filterKernel`** (3D) — apply the decision in place:

```
if !M[i] return
lab = find(parent, i)
keep = keepLargestOnly ? (lab == largestLabel) : (counts[lab] >= minSize)
if !keep: M[i] = 0
```

### Result
`CclResult` reports `numComponentsTotal`, `numComponentsKept`,
`largestComponent`, `foregroundBefore`, `foregroundAfter`, `iterations`.

### Correctness argument
Two voxels share a root **iff** a 26-connected foreground path joins them: the
merge step unions exactly the edges of the foreground adjacency graph, and
`atomicMin` collapses each connected subgraph to its minimum index. Compression
changes only path lengths, never roots. The loop stops when no root changes, i.e.
at the fix-point of the union operations.

### Cost
- Time: `O(K · N · 13)` for merge/compress (K = iterations, small) plus `O(N)`
  histogram/stats. Memory-bound.
- Memory: `parent` + `counts` = `8·N` bytes (≈ 67 MB for 8.39 M voxels).

---

## Stage 3 — 3D morphological closing + opening

**File:** [`morphology.cu`](../src/morphology.cu) · **Entry:** `morphOpenClose()`

### Purpose
- **Closing** (dilate → erode): bridges **inter-layer discontinuities** (a slice
  where the mask momentarily dropped out) and **fills interior holes**.
- **Opening** (erode → dilate): removes **burrs / thin spikes** and other small
  protrusions on the surface.

The pipeline runs **closing first, then opening** so gaps are mended before loose
fragments are trimmed.

### Binary morphology as min/max filtering
On a `{0,1}` volume with structuring element (SE) `B`:

\[ \text{erode}(M)(p) = \min_{o \in B} M(p+o), \qquad \text{dilate}(M)(p) = \max_{o \in B} M(p+o) \]

### Separable box SE
We use a **box** SE with per-axis radii `(rx, ry, rz)`. A box is the Minkowski sum
of three 1-D segments, and min/max filters are **separable**, so a 3-D box
erosion equals three successive 1-D erosions along X, then Y, then Z. This turns
an `O(rx·ry·rz)` neighbourhood into `O(rx+ry+rz)`.

**`morphAxisKernel`** — 1-D min (erode) or max (dilate) along one axis:

```
acc = dilate ? 0 : 1
for d in [-r, +r]:
    (nx,ny,nz) = clamp the sample coordinate along `axis`   # replicate padding
    v = in[idx]
    acc = dilate ? max(acc, v) : min(acc, v)
out[voxel] = acc
```

**`separable(src, A, B, dst)`** routes the three passes `X: src→A`, `Y: A→B`,
`Z: B→dst` (requires `dst ≠ B`; `dst` may equal `src` or `A`). `morphOpenClose`
ping-pongs two scratch buffers `t1, t2` so no extra allocation is needed:

```
close: separable(mask, t1, t2, t1, dilate);  separable(t1, t2, t1, mask, erode)
open : separable(mask, t1, t2, t1, erode);   separable(t1, t2, t1, mask, dilate)
```

### Bridging inter-layer gaps
Because the SE radii are **per-axis**, raising `rz` elongates the SE along the
slice direction (z). A single empty slice between two filled slabs is closed when
`rz ≥ 1`: dilation makes the neighbouring slabs overlap across the gap, and the
following erosion keeps the bridged voxel (both its z-neighbours are foreground).
Use `rz = 2` to bridge a two-slice gap, etc.

### Important subtlety (verified by the self-test)
A **1-D** opening along a single axis cannot remove a burr that lies *along that
axis*: erosion may drop the spike tip, but the surviving neighbour is re-dilated
back into it (a length-5 run is already "open" w.r.t. a size-3 SE). Removing thin
spikes therefore requires a **3-D box** SE (`rx = ry = rz ≥ 1`): erosion deletes
the whole 1-voxel-thick spike (its 3×3×3 neighbourhood is not all foreground) and
dilation cannot regenerate it. This is why the default radii are `1 1 1`.

### Cost
- Time: `O(N · (rx+ry+rz))` per erosion/dilation; open+close = 4 ops × 3 passes = **12 kernel launches**.
- Memory: `2·N` bytes scratch (`t1, t2`).

---

## Stage 4 — 3D Gaussian smoothing + threshold

**File:** [`gaussian.cu`](../src/gaussian.cu) · **Entry:** `gaussianSmoothMask()`

### Purpose
Low-pass filter the mask to **suppress edge noise** (single-voxel jitter, specks
on the surface) and produce a smooth, regular boundary — while thresholding keeps
the result a clean binary mask.

### Separable Gaussian
The 3-D Gaussian kernel factorizes:

\[ G(x,y,z) = g(x)\,g(y)\,g(z), \qquad g(d) = \exp\!\left(-\frac{d^2}{2\sigma^2}\right) \]

so a 3-D convolution becomes three 1-D convolutions.

### Kernel construction (host, uploaded once)
- radius `kr = max(1, ⌈3σ⌉)`, length `klen = 2·kr + 1`;
- weights `w[d] = exp(-d²/(2σ²))`, normalized so `Σ w = 1`.

### Kernel sequence
```
toFloatKernel : f1[i] = M[i] ? 1.0 : 0.0                       # uint8 → float
gaussAxis X   : f1 → f2                                         # 1-D conv along x
gaussAxis Y   : f2 → f1
gaussAxis Z   : f1 → f2
thresholdKernel: M[i] = (f2[i] >= threshold) ? 1 : 0            # default threshold 0.5
```
`gaussAxisKernel` accumulates `Σ_d in[clamp(coord+d)] · kern[d+kr]` with replicate
padding. Work is done in `float` to avoid quantization, then re-binarized.

### Why it removes noise but keeps the organ
- An **isolated voxel** becomes, after the separable filter, a peak of about
  `w(0)³ ≈ 0.023` (for σ=1) — far below 0.5 — so it vanishes.
- The **interior** of the organ is surrounded by foreground within the kernel, so
  its smoothed value stays near 1 and survives.
- Thresholding at **0.5** is roughly volume-neutral on a symmetric kernel; raising
  the threshold shrinks the mask, lowering it grows the mask.

### Cost
- Time: `O(N · klen · 3)`.
- Memory: two float buffers `f1, f2` = `8·N` bytes total + the small kernel array.
- If `sigma ≤ 0`, the stage is skipped.

---

## Stage 5 — Finalize: volume statistics and output

**File:** [`main.cu`](../src/main.cu) (with `countForegroundDevice` in [`volume.cu`](../src/volume.cu))

### Steps
1. **D2H** copy `d_mask → h_out`.
2. **Foreground count on GPU** (`countForegroundDevice`): a reduction kernel does
   `atomicAdd(&counter, 1)` for every foreground voxel (counter is
   `unsigned long long`) — avoids a slow host loop over `N`.
3. **Volume statistics**:

   \[ V_{\text{mm}^3} = \text{finalFg} \cdot dx \cdot dy \cdot dz, \quad
      V_{\text{cm}^3} = V_{\text{mm}^3}/1000, \quad
      V_{\text{L}} = V_{\text{mm}^3}/10^{6} \]

4. **Write outputs**:
   - `<prefix>_final.u8` — the cleaned binary mask (raw `uint8`, `N` bytes),
     directly loadable in 3D Slicer / ITK-SNAP / ParaView (with dims+spacing from
     `meta.txt`) for **3D visualization**;
   - `<prefix>_preview_z###.pgm` — evenly spaced axial previews at
     `z = p·(D-1)/(np-1)`.
5. **Optional evaluation**: Dice vs. a ground-truth stack,
   \( \text{Dice} = \frac{2|A \cap B|}{|A| + |B|} \), reported for the
   reconstructed and final masks (evaluation only — not part of the pipeline).

---

## Timing methodology

**Timer:** `Stopwatch` ([`common.cuh`](../src/common.cuh)) uses
`std::chrono::high_resolution_clock` and calls `cudaDeviceSynchronize()` before
reading the stop time, so each measurement is true wall-clock including kernel
launches, device execution, and scratch `cudaMalloc`/`cudaFree`.

**Making the numbers trustworthy:**
- **Warm-up pass** — one untimed run of the whole chain ramps the GPU from its
  idle power state and primes the caching allocator (otherwise the first run is
  dominated by clock ramp-up and first-touch allocation).
- **Repeat & average** — GPU stages are run `--repeat R` times (default 5) and
  averaged. Between repeats the volume is reset from a pristine device copy
  `d_recon` (an untimed D2D copy) so every repeat starts from the same state.
- **Excluded** from stage timings: warm-up, the reset copy, host GT load, printing.

### Measured results — RTX 5070 Ti, 256×256×128 (8.39 M voxels), mean of 5

| Step | Time | Notes |
|---|---:|---|
| 1a. load slices (host) | ~0.93 ms | file I/O |
| 1b. H2D + binarize | ~0.69 ms | PCIe upload + `O(N)` kernel |
| 2. 3D CCL + filter | ~1.22 ms | 174 → 1 components, 3 iterations |
| 3. morphology (close+open) | ~0.77 ms | 12 separable passes |
| 4. Gaussian + threshold | ~0.82 ms | σ=1.0 |
| 5. finalize (D2H+stats+write) | ~8.68 ms | **I/O-bound** (8 MB write) |
| **TOTAL** | **~13.1 ms** | |

Pure GPU compute (1b–4) is only a few milliseconds; the final step is dominated
by writing the mask to disk. Quality on the synthetic UNet-like prediction:
**Dice 0.9845 → 0.9876 (+0.0031)**, organ volume ≈ **1777.7 cm³**.

---

## Correctness verification

**Kernel self-test** (`./bin/liver3d --selftest`, [`selftest.cu`](../src/selftest.cu))
uses tiny volumes with hand-computed answers:
- **CCL:** a 4×4×4 block (64 voxels) + a separate 2-voxel speck → `found=2`,
  `kept=1` (minSize=10), speck cleared, block retained.
- **Closing:** two slabs separated by an empty slice `z=3` → the gap voxel is
  filled (inter-layer bridge).
- **Opening:** a block with a thin 1-voxel spike → spike removed, block kept
  (demonstrates the 3-D-box requirement above).
- **Gaussian:** an 8³ block + an isolated voxel → block interior kept, isolated
  voxel suppressed.

**Integration test** ([`../tests/test_pipeline.py`](../tests/test_pipeline.py))
runs the binary and asserts: all steps reported timings, CCL reduced the
component count, the output is strictly binary, and its voxel count / volume /
Dice match the report with no accuracy regression.

---

## Parameter tuning guide (liver masks)

| Parameter | Default | Guidance |
|---|---|---|
| `--bin` | 0 | input foreground threshold; keep 0 for `0/1` masks, use 127 for `0/255` |
| `--minsize` | 64 | remove components smaller than this. Liver ≈ 7·10⁵ voxels, FPs ≪ 100 → 64 is safe; raise to be more aggressive |
| `--keep-largest` | off | keep only the single largest component (strictest FP removal) |
| `--mrx/--mry` | 1 | in-plane smoothing / burr removal |
| `--mrz` | 1 | **increase to bridge thicker inter-layer gaps** (2 → two-slice gaps) |
| `--no-close`/`--no-open` | off | disable one morphological direction |
| `--sigma` | 1.0 | edge smoothing strength (0.8–1.5 typical) |
| `--gthr` | 0.5 | re-binarize level; >0.5 shrinks, <0.5 grows the mask |

---

## Complexity & memory summary

| Stage | Time | Device memory | Launches |
|---|---|---|---|
| 1 reconstruct | `O(N)` + I/O | `2·N` (transient) | 1 |
| 2 CCL + filter | `O(K·N·13) + O(N)` | `8·N` (parent+counts) | `2K + 3` |
| 3 morphology | `O(N·(rx+ry+rz))` | `2·N` (scratch) | 12 |
| 4 Gaussian | `O(N·klen·3)` | `8·N` (2 float buffers) | 5 |
| 5 finalize | `O(N)` + I/O | small (counter) | 1 |

(`N` bytes = 1 byte/voxel; `4·N` = `int`; `8·N` = `float`/two buffers. For
8.39 M voxels: `N ≈ 8 MB`, `8·N ≈ 67 MB`.)

---

## Extending the pipeline
- **Mesh export for 3D visualization:** add a marching-cubes stage on `d_mask` to
  emit STL/PLY/OBJ instead of (or in addition to) the raw volume.
- **Real UNet results:** convert predictions with
  [`../tools/import_masks.py`](../tools/import_masks.py) (PNG/PGM/PBM/RAW →
  `slices/*.u8` + `meta.txt`). Public liver CT sources: **3D-IRCADb-01**, **LiTS**.
- **Signed-distance / float masks:** replace `binarizeKernel`'s threshold to
  ingest probability maps directly.
