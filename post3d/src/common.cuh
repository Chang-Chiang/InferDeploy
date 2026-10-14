#pragma once

/*
 * 通用 CUDA 辅助设施：错误检查 + 墙钟计时器。
 */
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

/// @brief 任何 CUDA 运行时调用失败时，打印文件/行号与错误描述后直接退出
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n  -> %s\n", __FILE__, __LINE__, #call, \
                    cudaGetErrorString(err__));                                             \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

/// @brief 便捷写法：为尺寸 (W,H,D) 的三维体数据推导 grid 维度
inline dim3 gridFor(int W, int H, int D, dim3 block = dim3(8, 8, 8)) {
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);
    return grid;
}

/// @brief 用于逐步耗时上报的墙钟秒表。读数之前会同步设备，所以测得的时间
///        包含核函数启动、设备执行，以及单个步骤内部的临时显存分配
struct Stopwatch {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;

    void start() {
        t0 = clk::now();
    }

    double stopMs() {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
