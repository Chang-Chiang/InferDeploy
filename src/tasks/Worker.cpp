#include <memory>
#include "include/tasks/Worker.h"
#include "include/common/ModelCfg.h"

/// @brief 按参数创建推理后端（TensorRT / OpenVINO）
static std::shared_ptr<InferBackend> CreateBackend(const Params& cParams) {
#ifdef ENABLE_OPENVINO
    if (cParams.m_eBackend == OPENVINO) {
        return std::make_shared<OvBackend>();
    }
#else
    if (cParams.m_eBackend == OPENVINO) {
        LOGE("OpenVINO backend is not compiled in! Rebuild with ENABLE_OPENVINO=1 (see config/Makefile.config). "
             "Fallback to TensorRT.");
    }
#endif
    return std::make_shared<TrtBackend>();
}

Worker::Worker(const std::string& strModelPath, Level cLevel, const Params& cParams)
    : m_cParams(cParams) {

    auto p_backend = CreateBackend(cParams);

    if (cParams.m_eTask == CLASSIFICATION) {
        m_pYoloClassifier = MakeClassifier(strModelPath, cLevel, cParams, p_backend);
    }
    else if (cParams.m_eTask == DETECTION) {
        m_pYoloDetector = MakeDetector(strModelPath, cLevel, cParams, p_backend);
    }
    else if (cParams.m_eTask == MULTITASK) {
        m_pYoloMultitask = MakeMultitask(strModelPath, cLevel, cParams, p_backend);
    }
    else if (cParams.m_eTask == SEGMENTATION) {
        m_pUnetSegmenter = MakeUnetSegmenter(strModelPath, cLevel, cParams, p_backend);
    }
}

void Worker::Inference(const std::string& strImagePath) {
    if (m_pYoloClassifier != nullptr) {
        m_pYoloClassifier->InitModel();
        m_pYoloClassifier->LoadImage(strImagePath);
        m_pYoloClassifier->Inference();
    }
    else if (m_pYoloDetector != nullptr) {
        m_pYoloDetector->InitModel();
        m_pYoloDetector->LoadImage(strImagePath);
        m_pYoloDetector->Inference();
    }
    else if (m_pYoloMultitask != nullptr) {
        m_pYoloMultitask->InitModel();
        m_pYoloMultitask->LoadImage(strImagePath);
        m_pYoloMultitask->Inference();
    }
    else if (m_pUnetSegmenter != nullptr) {
        m_pUnetSegmenter->InitModel();
        m_pUnetSegmenter->LoadImage(strImagePath);
        m_pUnetSegmenter->Inference();
    }
}

std::shared_ptr<Worker> CreateWorker(const std::string& strModelPath, Level cLevel, const Params& cParams) {
    return std::make_shared<Worker>(strModelPath, cLevel, cParams);
}
