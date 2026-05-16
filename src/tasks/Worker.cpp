#include "include/tasks/Worker.h"
#include "include/common/ModelCfg.h"

Worker::Worker(std::string strModelPath, Level cLevel, Params cParams) {
    m_pLogger = CreateLogger(cLevel);

    if (cParams.m_eTask == CLASSIFICATION) {
        m_pYoloClassifier = MakeClassifier(strModelPath, cLevel, cParams);
    }
    else if (cParams.m_eTask == DETECTION) {
        m_pYoloDetector = MakeDetector(strModelPath, cLevel, cParams);
    }
    else if (cParams.m_eTask == MULTITASK) {
        m_pYoloMultitask = MakeMultitask(strModelPath, cLevel, cParams);
    }
}

void Worker::Inference(std::string strImagePath) {
    if (m_pYoloClassifier != nullptr) {
        m_pYoloClassifier->InitModel();
        m_pYoloClassifier->LoadImage(strImagePath);
        m_pYoloClassifier->Inference();
    }

    if (m_pYoloDetector != nullptr) {
        m_pYoloDetector->InitModel();
        m_pYoloDetector->LoadImage(strImagePath);
        m_pYoloDetector->Inference();
    }

    if (m_pYoloMultitask != nullptr) {
        m_pYoloMultitask->InitModel();
        m_pYoloMultitask->LoadImage(strImagePath);
        m_pYoloMultitask->Inference();
    }
}

std::shared_ptr<Worker> CreateWorker(std::string strModelPath, Level cLevel, Params cParams) {
    return std::make_shared<Worker>(strModelPath, cLevel, cParams);
}