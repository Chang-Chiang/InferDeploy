#pragma once

#include "include/tasks/YoloClassifier.h"
#include "include/tasks/YoloDetector.h"

class Worker {
public:
    Worker(std::string strOnnxPath, Level cLevel, Params cParams);
    void Inference(std::string strImagePath);

public:
    std::shared_ptr<Logger>          m_pLogger;
    std::shared_ptr<Params>          m_pParams;
    std::shared_ptr<YoloClassifier>  m_pYoloClassifier;
    std::shared_ptr<YoloDetector>    m_pYoloDetector;
    std::vector<float>               m_vecScore;
    std::vector<Bbox>                m_vecBbox;
};

std::shared_ptr<Worker> CreateWorker(std::string strOnnxPath, Level cLevel, Params cParams);