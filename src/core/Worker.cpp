#include "include/core/Worker.h"
#include "include/common/ModelCfg.h"

Worker::Worker(std::string strModelPath, Level cLevel, Params cParams) {
    m_pLogger = CreateLogger(cLevel);

    if (cParams.m_eTtask == CLASSIFICATION) {
        m_pClassifier = MakeClassifier(strModelPath, cLevel, cParams);
    }
    else if (cParams.m_eTtask == DETECTION) {
        m_pDetector = MakeDetector(strModelPath, cLevel, cParams);
    }
}

void Worker::Inference(std::string strImagePath) {
    if (m_pClassifier != nullptr) {
        m_pClassifier->InitModel();
        m_pClassifier->LoadImage(strImagePath);
        m_pClassifier->inference();
    }

    if (m_pDetector != nullptr) {
        m_pDetector->InitModel();
        m_pDetector->LoadImage(strImagePath);
        m_pDetector->Inference();
    }
}

shared_ptr<Worker> CreateWorker(std::string strModelPath, Level cLevel, Params cParams) {
    return std::make_shared<Worker>(strModelPath, cLevel, cParams);
}