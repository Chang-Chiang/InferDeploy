#pragma once

/*
 * 体数据重建（逐切面掩码 -> 三维体）与 host 侧文件 I/O。
 */
#include <cuda_runtime.h>

#include <cstdint>

/// @brief 体数据元信息：尺寸（体素）与体素间距（mm）
struct VolumeMeta {
    int   W = 0, H = 0, D = 0;           // 宽、高、切面数（深度）
    float dx = 1.f, dy = 1.f, dz = 1.f;  // 体素间距，单位 mm

    size_t voxels() const {
        return (size_t)W * H * D;
    }

    float voxelVolumeMm3() const {
        return dx * dy * dz;
    }
};

/// @brief 从小文本文件读取 "W H D dx dy dz"
bool loadMeta(const char* path, VolumeMeta& m);

/// @brief 把逐切面裸数据文件 <dir>/slice_%04d.u8（每个 W*H 字节）拼成一个三维栈
bool loadSlicesDir(const char* dir, const VolumeMeta& m, uint8_t* hostBuf);

/// @brief 读 / 写一整块连续的 n 字节裸数据栈
bool loadStack(const char* path, size_t n, uint8_t* buf);
bool saveStack(const char* path, const uint8_t* buf, size_t n);

/// @brief 把一个轴位切面写成 P5 格式的 PGM 图像（0/1 拉伸到 0/255）
bool savePGM(const char* path, const uint8_t* slice, int W, int H);

/// @brief GPU 侧重建核：把裸字节栈拷到设备并二值化（值 > binThreshold 即前景）
/// @return 设备指针，由调用方负责释放
uint8_t* reconstructOnDevice(const uint8_t* h_raw, const VolumeMeta& m, int binThreshold);

/// @brief host 侧辅助：统计一段数据栈中的前景体素数
long countForeground(const uint8_t* buf, size_t n);

/// @brief 设备侧辅助：统计设备上掩码的前景体素数（GPU 归约）
long countForegroundDevice(const uint8_t* d_mask, size_t n);
