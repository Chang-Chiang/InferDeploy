#include "include/tasks/YoloClassifier.h"
#include "include/tasks/ImageNetLabels.hpp"
#include "include/utils/Process.h"
#include "include/utils/CudaCheck.h"

#include <memory>

void YoloClassifier::SetupIoBuffers() {
    m_nvInputDims  = m_vecInputs[0].dims;
    m_nvOutputDims = m_vecOutputs[0].dims;

    m_iInputSize = m_cParams.m_cImgInfo.m_iHeight * m_cParams.m_cImgInfo.m_iWidth * m_cParams.m_cImgInfo.m_iChannel *
                   sizeof(float);
    m_iOutputSize = m_cParams.m_iNumCls * sizeof(float);
    m_iImgArea    = m_cParams.m_cImgInfo.m_iHeight * m_cParams.m_cImgInfo.m_iWidth;

    CUDA_CHECK(cudaMallocHost((void**)&m_pInputMemory[0], m_iInputSize));
    CUDA_CHECK(cudaMallocHost((void**)&m_pOutputMemory[0], m_iOutputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pInputMemory[1], m_iInputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pOutputMemory[1], m_iOutputSize));
}

void YoloClassifier::ResetTask() {
}

bool YoloClassifier::PreprocessCpu() {
    // 数据集三通道均值、方差
    float arr_mean[] = {0.406, 0.456, 0.485};
    float arr_std[]  = {0.225, 0.224, 0.229};

    cv::Mat mat_input_image;
    mat_input_image = cv::imread(m_strImagePath);
    if (mat_input_image.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated");
        return false;
    }

    m_cTimer.StartCpu();

    cv::resize(mat_input_image, mat_input_image,
               cv::Size(m_cParams.m_cImgInfo.m_iWidth, m_cParams.m_cImgInfo.m_iHeight), 0, 0, cv::INTER_LINEAR);

    // normalize and convert to CHW
    int i_idx;
    int i_offset_ch0 = m_iImgArea * 0;
    int i_offset_ch1 = m_iImgArea * 1;
    int i_offset_ch2 = m_iImgArea * 2;
    for (int i = 0; i < m_nvInputDims.d[2]; i++) {
        for (int j = 0; j < m_nvInputDims.d[3]; j++) {
            i_idx                             = i * m_nvInputDims.d[3] * m_nvInputDims.d[1] + j * m_nvInputDims.d[1];
            m_pInputMemory[0][i_offset_ch2++] = (mat_input_image.data[i_idx + 0] / 255.0f - arr_mean[0]) / arr_std[0];
            m_pInputMemory[0][i_offset_ch1++] = (mat_input_image.data[i_idx + 1] / 255.0f - arr_mean[1]) / arr_std[1];
            m_pInputMemory[0][i_offset_ch0++] = (mat_input_image.data[i_idx + 2] / 255.0f - arr_mean[2]) / arr_std[2];
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_pInputMemory[1], m_pInputMemory[0], m_iInputSize,
                               cudaMemcpyKind::cudaMemcpyHostToDevice, m_cuStream));

    m_cTimer.StopCpu<Timer::ms>("preprocess(CPU)");

    return true;
}

bool YoloClassifier::PreprocessGpu() {
    // 数据集三通道均值、标准差
    float arr_mean[] = {0.406, 0.456, 0.485};
    float arr_std[]  = {0.225, 0.224, 0.229};

    cv::Mat mat_input_image;
    mat_input_image = cv::imread(m_strImagePath);
    if (mat_input_image.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated");
        return false;
    }

    m_cTimer.StartGpu();

    ResizeGpu(mat_input_image, m_pInputMemory[1], m_cParams.m_cImgInfo.m_iHeight, m_cParams.m_cImgInfo.m_iWidth,
              arr_mean, arr_std, InterpolationTactics::GPU_BILINEAR);

    m_cTimer.StopGpu("preprocess(GPU)");
    return true;
}

bool YoloClassifier::PostprocessCpu() {

    m_cTimer.StartCpu();

    int i_output_size = m_cParams.m_iNumCls * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_pOutputMemory[0], m_pOutputMemory[1], i_output_size,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    ImageNetLabels c_labels;
    int   i_pos = std::max_element(m_pOutputMemory[0], m_pOutputMemory[0] + m_cParams.m_iNumCls) - m_pOutputMemory[0];
    float f_confidence = m_pOutputMemory[0][i_pos] * 100;

    // 保存分类结果，供外部（如量化精度评估工具）读取
    m_iTop1Index  = i_pos;
    m_fConfidence = m_pOutputMemory[0][i_pos];

    m_cTimer.StopCpu<Timer::ms>("postprocess(CPU)");

    LOG("Result:     %s", c_labels.ImagenetLabelString(i_pos).c_str());
    LOG("Confidence  %.3f%%", f_confidence);

    return true;
}

std::unique_ptr<YoloClassifier> MakeClassifier(const std::string& strModelPath, Level cLevel, const Params& cParams,
                                               std::shared_ptr<InferBackend> pBackend) {
    return std::make_unique<YoloClassifier>(strModelPath, cLevel, cParams, std::move(pBackend));
}
