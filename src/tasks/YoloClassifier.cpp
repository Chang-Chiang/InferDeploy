#include "include/tasks/YoloClassifier.h"
#include "include/tasks/ImageNetLabels.hpp"
#include "include/utils/Process.h"

#include <memory>
void YoloClassifier::Setup(void const* pData, size_t iSize) {
    m_pRuntime     = shared_ptr<IRuntime>(createInferRuntime(*m_pLogger), DestroyTrtPtr<IRuntime>);
    m_pEngine      = shared_ptr<ICudaEngine>(m_pRuntime->deserializeCudaEngine(pData, iSize), DestroyTrtPtr<ICudaEngine>);
    m_pContext     = shared_ptr<IExecutionContext>(m_pEngine->createExecutionContext(), DestroyTrtPtr<IExecutionContext>);
    m_nvInputDims  = m_pContext->getBindingDimensions(0);
    m_nvOutputDims = m_pContext->getBindingDimensions(1);

    CUDA_CHECK(cudaStreamCreate(&m_cuStream));
    
    m_iInputSize  = m_pParams->m_cImgInfo.m_iHeight * m_pParams->m_cImgInfo.m_iWidth * m_pParams->m_cImgInfo.m_iChannel * sizeof(float);
    m_iOutputSize = m_pParams->m_iNumCls * sizeof(float);
    m_iImgArea    = m_pParams->m_cImgInfo.m_iHeight * m_pParams->m_cImgInfo.m_iWidth;

    CUDA_CHECK(cudaMallocHost(&m_pInputMemory[0], m_iInputSize));
    CUDA_CHECK(cudaMallocHost(&m_pOutputMemory[0], m_iOutputSize));
    CUDA_CHECK(cudaMalloc(&m_pInputMemory[1], m_iInputSize));
    CUDA_CHECK(cudaMalloc(&m_pOutputMemory[1], m_iOutputSize));

    m_pBindings[0] = m_pInputMemory[1];
    m_pBindings[1] = m_pOutputMemory[1];
}

void YoloClassifier::ResetTask() {}

bool YoloClassifier::PreprocessCpu() {

    float arr_mean[] = {0.406, 0.456, 0.485};
    float arr_std[]  = {0.225, 0.224, 0.229};

    cv::Mat mat_input_image;
    mat_input_image = cv::imread(m_strImagePath);
    if (mat_input_image.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    m_pTimer->StartCpu();

    cv::resize(mat_input_image, mat_input_image, 
               cv::Size(m_pParams->m_cImgInfo.m_iWidth, m_pParams->m_cImgInfo.m_iHeight), 0, 0, cv::INTER_LINEAR);

    int i_idx;
    int i_offset_ch0 = m_iImgArea * 0;
    int i_offset_ch1 = m_iImgArea * 1;
    int i_offset_ch2 = m_iImgArea * 2;
    for (int i = 0; i < m_nvInputDims.d[2]; i++) {
        for (int j = 0; j < m_nvInputDims.d[3]; j++) {
            i_idx = i * m_nvInputDims.d[3] * m_nvInputDims.d[1] + j * m_nvInputDims.d[1];
            m_pInputMemory[0][i_offset_ch2++] = (mat_input_image.data[i_idx + 0] / 255.0f - arr_mean[0]) / arr_std[0];
            m_pInputMemory[0][i_offset_ch1++] = (mat_input_image.data[i_idx + 1] / 255.0f - arr_mean[1]) / arr_std[1];
            m_pInputMemory[0][i_offset_ch0++] = (mat_input_image.data[i_idx + 2] / 255.0f - arr_mean[2]) / arr_std[2];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_pInputMemory[1], m_pInputMemory[0], m_iInputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_cuStream));

    m_pTimer->StopCpu<Timer::ms>("preprocess(CPU)");

    return true;
}

bool YoloClassifier::PreprocessGpu() {

    float arr_mean[] = {0.406, 0.456, 0.485};
    float arr_std[]  = {0.225, 0.224, 0.229};

    cv::Mat mat_input_image;
    mat_input_image = cv::imread(m_strImagePath);
    if (mat_input_image.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }

    m_pTimer->StartGpu();

    ResizeGpu(mat_input_image, m_pInputMemory[1],
              m_pParams->m_cImgInfo.m_iHeight, m_pParams->m_cImgInfo.m_iWidth,
              arr_mean, arr_std,
              InterpolationTactics::GPU_BILINEAR);

    m_pTimer->StopGpu("preprocess(GPU)");
    return true;
}


bool YoloClassifier::PostprocessCpu() {

    m_pTimer->StartCpu();

    int i_output_size = m_pParams->m_iNumCls * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_pOutputMemory[0], m_pOutputMemory[1], i_output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    ImageNetLabels c_labels;
    int i_pos = std::max_element(m_pOutputMemory[0], m_pOutputMemory[0] + m_pParams->m_iNumCls) - m_pOutputMemory[0];
    float f_confidence = m_pOutputMemory[0][i_pos] * 100;

    m_pTimer->StopCpu<Timer::ms>("postprocess(CPU)");

    LOG("Result:     %s", c_labels.ImagenetLabelString(i_pos).c_str());   
    LOG("Confidence  %.3f%%", f_confidence);   

    m_pTimer->show();
    printf("\n");

    return true;
}


bool YoloClassifier::PostprocessGpu() {
    return PostprocessCpu();
}

std::shared_ptr<YoloClassifier> MakeClassifier(std::string strModelPath, Level cLevel, Params cParams) {
    return make_shared<YoloClassifier>(strModelPath, cLevel, cParams);
}