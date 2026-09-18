#pragma once
// Volume reconstruction (slice-by-slice masks -> 3D body) and host-side I/O.
#include <cstdint>
#include <cuda_runtime.h>

struct VolumeMeta {
    int   W = 0, H = 0, D = 0;   // width, height, number of slices (depth)
    float dx = 1.f, dy = 1.f, dz = 1.f;  // voxel spacing in mm
    size_t voxels() const { return (size_t)W * H * D; }
    float  voxelVolumeMm3() const { return dx * dy * dz; }
};

// Read "W H D dx dy dz" from a small text file.
bool loadMeta(const char* path, VolumeMeta& m);

// Assemble a 3D stack from per-slice raw files  <dir>/slice_%04d.u8  (W*H bytes each).
bool loadSlicesDir(const char* dir, const VolumeMeta& m, uint8_t* hostBuf);

// Read / write a single contiguous raw stack of n bytes.
bool loadStack(const char* path, size_t n, uint8_t* buf);
bool saveStack(const char* path, const uint8_t* buf, size_t n);

// Write one axial slice as a P5 PGM image (values 0/1 scaled to 0/255).
bool savePGM(const char* path, const uint8_t* slice, int W, int H);

// GPU reconstruction kernel: copy raw byte stack to device and binarize
// (foreground where value > binThreshold). Returns device pointer (caller frees).
uint8_t* reconstructOnDevice(const uint8_t* h_raw, const VolumeMeta& m,
                             int binThreshold);

// Host helper: count foreground voxels in a stack.
long countForeground(const uint8_t* buf, size_t n);

// Device helper: count foreground voxels in a device mask (GPU reduction).
long countForegroundDevice(const uint8_t* d_mask, size_t n);
