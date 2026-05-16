#pragma once

#include <memory>
#include "include/tasks/YoloClassifier.h"
#include "include/tasks/YoloDetector.h"
#include "include/tasks/YoloMultitask.h"

class Worker {
public:
    /// @brief 构造函数，负责根据任务类型创建对应的推理对象
    /// @param strOnnxPath ONNX 模型路径
    /// @param cLevel 日志级别
    /// @param cParams 模型构建参数
    Worker(std::string strOnnxPath, Level cLevel, Params cParams);
    
    /// @brief 推理函数，根据任务类型进行推理
    /// @param strImagePath 待推理的图像路径
    /// @return 无返回值
    void Inference(std::string strImagePath);

public:
    Params                           m_cParams;
    std::unique_ptr<YoloClassifier>  m_pYoloClassifier;
    std::unique_ptr<YoloDetector>    m_pYoloDetector;
    std::unique_ptr<YoloMultitask>   m_pYoloMultitask;
    std::vector<float>               m_vecScore;
    std::vector<Bbox>                m_vecBbox;
};

std::shared_ptr<Worker> CreateWorker(std::string strOnnxPath, Level cLevel, Params cParams);