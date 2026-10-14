#include "include/core/InferBase.h"

#include "include/utils/FileUtil.h"
#include "include/utils/CudaCheck.h"

InferBase::InferBase(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                     std::shared_ptr<InferBackend> pBackend)
    : m_cLogger(cLevel)
    , m_cParams(cParams)
    , m_strOnnxPath(strOnnxPath)
    , m_cuStream(nullptr)
    , m_pBackend(std::move(pBackend)) {
    m_pInputMemory[0]  = nullptr;
    m_pInputMemory[1]  = nullptr;
    m_pOutputMemory[0] = nullptr;
    m_pOutputMemory[1] = nullptr;
    CUDA_CHECK(cudaStreamCreate(&m_cuStream));
}

InferBase::~InferBase() {
    ReleaseExtraBuffers();
    CleanupBuffers();
    if (m_pBackend) {
        m_pBackend->Release();
    }
}

void InferBase::CleanupBuffers() {
    if (m_pInputMemory[0]) {
        cudaFreeHost(m_pInputMemory[0]);
        m_pInputMemory[0] = nullptr;
    }
    if (m_pInputMemory[1]) {
        cudaFree(m_pInputMemory[1]);
        m_pInputMemory[1] = nullptr;
    }
    if (m_pOutputMemory[0]) {
        cudaFreeHost(m_pOutputMemory[0]);
        m_pOutputMemory[0] = nullptr;
    }
    if (m_pOutputMemory[1]) {
        cudaFree(m_pOutputMemory[1]);
        m_pOutputMemory[1] = nullptr;
    }
    if (m_cuStream) {
        cudaStreamDestroy(m_cuStream);
        m_cuStream = nullptr;
    }
}

void InferBase::LoadImage(const std::string& strImagePath) {
    if (!IsFileExist(strImagePath)) {
        LOGE("%s not found", strImagePath.c_str());
    }
    else {
        m_strImagePath = strImagePath;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", GetFileName(m_strOnnxPath).c_str());
        LOG("\tImage:      %s", GetFileName(m_strImagePath).c_str());
        LOG("\tPrecision:  %s", GetPrec(m_cParams.m_ePrec).c_str());
    }
}

void InferBase::InitModel() {
    if (m_vecInputs.empty()) {
        // 首次初始化：加载模型 → 任务类分配缓冲 → 后端绑定缓冲
        if (!m_pBackend->LoadModel(this)) {
            LOGE("LoadModel failed! backend: %s", m_cParams.m_eBackend == OPENVINO ? "OpenVINO" : "TensorRT");
            return;
        }
        SetupIoBuffers();
        if (!m_pBackend->BindBuffers(this)) {
            LOGE("BindBuffers failed!");
            return;
        }
        LOG("%s model initialized. backend: %s", GetFileName(m_strOnnxPath).c_str(),
            m_cParams.m_eBackend == OPENVINO ? "OpenVINO" : "TensorRT");
    }
    else {
        m_cTimer.Init();
        ResetTask();
    }
}

void InferBase::Inference() {
    if (m_cParams.m_eDevice == CPU) {
        PreprocessCpu();
    }
    else {
        PreprocessGpu();
    }

    if (!m_pBackend->Enqueue(this)) {
        LOG("Error happens during DNN inference part, program terminated");
        return;
    }

    if (m_cParams.m_eDevice == CPU) {
        PostprocessCpu();
    }
    else {
        PostprocessGpu();
    }
}

std::string InferBase::GetPrec(Precision cPrec) {
    switch (cPrec) {
        case Precision::FP16:
            return "fp16";
        case Precision::INT8:
            return "int8";
        default:
            return "fp32";
    }
}
