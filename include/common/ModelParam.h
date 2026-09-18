#pragma once

#include "include/utils/Process.h"
#include <string>
#include <vector>

// ONNX 解析器设置的 临时内存上限，用于 将 ONNX 模型转成 TRT 网络
// 256 MB
#define WORKSPACESIZE (1 << 28)

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

/// @brief INT8 校准算法类别
enum CalibAlgo {
    ENTROPY,  // IInt8EntropyCalibrator2，基于信息熵，通用性强，但 YOLOv8 等 DWConv 较多架构上易掉点
    MINMAX    // IInt8MinMaxCalibrator，保留 FP32 极值分布，YOLOv8 上精度恢复更好
};

/// @brief 推理后端类别
enum InferBackendType {
    TENSORRT,   // NVIDIA TensorRT（GPU）
    OPENVINO    // Intel OpenVINO（CPU/GPU，需 ENABLE_OPENVINO 编译开关）
};

/// @brief 后端无关的张量维度描述（兼容 xxx.d[i] 访问习惯）
struct Dims {
    int nbDims = 0;
    int d[8]   = {};
    int operator[](int i) const { return d[i]; }
    int& operator[](int i)      { return d[i]; }
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
    int                  m_iCalibBatchSize      = 64;   // INT8 校准 batch size
    std::string          m_strCalibListPath     = "../calibration/calibration_list.txt";  // 校准数据列表路径
    CalibAlgo            m_eCalibAlgo           = ENTROPY;  // INT8 校准算法
    InferBackendType     m_eBackend             = TENSORRT; // 推理后端
    std::vector<std::string> m_vecFp32Layers;  // 需要强制 FP32 精度的层名列表
};