#pragma once
// 3D binary morphology with a separable box structuring element.
//   closing (dilate then erode): bridges inter-layer gaps, fills holes
//   opening (erode then dilate): removes burrs / thin spurious protrusions
// The box is separable, so each 3D op is three 1D min/max passes (O(r) cost
// per voxel instead of O(r^3)). Radii may differ per axis to elongate the
// structuring element along Z (slice direction).
#include <cstdint>

// Applies closing then opening (either can be disabled) to a 0/1 uint8 volume.
void morphOpenClose(uint8_t* d_mask, int W, int H, int D,
                    int rx, int ry, int rz,
                    bool doClose, bool doOpen);
