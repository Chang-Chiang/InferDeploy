#pragma once

#include "include/utils/Process.h"

// ONNX 解析器设置的 临时内存上限，用于 将 ONNX 模型转成 TRT 网络
// 256 MB
#define WORKSPACESIZE 1<<28

/// @brief 任务类别
enum TaskType {
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
    MULTITASK
};

/// @brief 设备类别
enum Device {
    CPU,
    GPU
};

/// @brief 精度类别
enum Precision {
    FP32,
    FP16,
    INT8
};

/// @brief 图像信息
struct ImageInfo {
    int m_iHeight;
    int m_iWidth;
    int m_iChannel;
    ImageInfo(int iHeight, int iWidth, int iChannel) 
        : m_iHeight(iHeight)
        , m_iWidth(iWidth)
        , m_iChannel(iChannel) {}
};

/// @brief 模型构建参数
struct Params {
    Device               m_eDevice              = GPU;
    int                  m_iNumCls              = 1000;
    InterpolationTactics m_eInterpolationTactic = InterpolationTactics::GPU_BILINEAR;
    ImageInfo            m_cImgInfo             = {224, 224, 3};
    TaskType             m_eTask                = CLASSIFICATION;
    int                  m_iWorkspaceSize       = WORKSPACESIZE;
    Precision            m_ePrec                = FP32;
};