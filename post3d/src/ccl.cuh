#pragma once
// 3D connected-component labeling (26-connectivity) on the GPU + size filtering.
// Removes inter-layer isolated false positives (small components).
#include <cstdint>

struct CclResult {
    int numComponentsTotal = 0;  // connected components found (size >= 1)
    int numComponentsKept  = 0;  // components surviving the size filter
    int largestComponent   = 0;  // voxel count of the biggest component
    long foregroundBefore  = 0;  // foreground voxels before filtering
    long foregroundAfter   = 0;  // foreground voxels after filtering
    int iterations         = 0;  // union-find merge/compress iterations to converge
};

// In-place: d_mask is a 0/1 uint8 volume of size W*H*D.
// Components with fewer than `minSize` voxels are cleared to background.
// If `keepLargestOnly` is true, only the single largest component survives.
CclResult filterComponents(uint8_t* d_mask, int W, int H, int D,
                           int minSize, bool keepLargestOnly);
