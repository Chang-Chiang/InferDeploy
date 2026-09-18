#pragma once
// Common CUDA helpers: error checking and a wall-clock / GPU-event timer.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Abort with a descriptive message on any CUDA runtime failure.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  -> %s\n", __FILE__,     \
                    __LINE__, #call, cudaGetErrorString(err__));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Convenience: pick a block/grid for a 3D volume of size (W,H,D).
inline dim3 gridFor(int W, int H, int D, dim3 block = dim3(8, 8, 8)) {
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (D + block.z - 1) / block.z);
    return grid;
}

// Wall-clock stopwatch used to report per-step timings. All GPU work is
// synchronized before reading, so the measured time includes kernel launches,
// device execution and any scratch allocation inside a step.
struct Stopwatch {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void start() { t0 = clk::now(); }
    double stopMs() {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
