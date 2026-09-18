#pragma once
// Separable 3D Gaussian smoothing of a binary mask, followed by thresholding.
// Converts the 0/1 mask to float, convolves with a 1D Gaussian along X, Y and Z
// (separable => O(k) per voxel), then re-thresholds at `threshold` (default 0.5)
// to obtain a regularized, edge-noise-suppressed binary mask.
#include <cstdint>

// In-place on d_mask. sigma in voxels; threshold in [0,1].
void gaussianSmoothMask(uint8_t* d_mask, int W, int H, int D,
                        float sigma, float threshold);
