#include "include/tasks/UnetSegmenter.h"
#include "include/utils/Process.h"
#include "include/utils/CudaCheck.h"
#include "include/utils/FileUtil.h"

#include <memory>

void UnetSegmenter::SetupIoBuffers() {
    m_nvInputDims  = m_vecInputs[0].dims;
    m_nvOutputDims = m_vecOutputs[0].dims;

    // 输入 1x1x256x256，输出 1x1x256x256（灰度单通道，UNet 保持分辨率）
    m_iInputSize  = m_nvInputDims.d[1] * m_nvInputDims.d[2] * m_nvInputDims.d[3] * sizeof(float);
    m_iOutputSize = m_nvOutputDims.d[1] * m_nvOutputDims.d[2] * m_nvOutputDims.d[3] * sizeof(float);

    CUDA_CHECK(cudaMallocHost((void**)&m_pInputMemory[0], m_iInputSize));
    CUDA_CHECK(cudaMallocHost((void**)&m_pOutputMemory[0], m_iOutputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pInputMemory[1], m_iInputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pOutputMemory[1], m_iOutputSize));

    // 同步 Params 中的图像尺寸（与其它任务的后端回写行为保持一致）
    m_cParams.m_cImgInfo = {m_nvInputDims.d[2], m_nvInputDims.d[3], m_nvInputDims.d[1]};
}

void UnetSegmenter::ResetTask() {
    m_matMask.release();
}

bool UnetSegmenter::PreprocessCpu() {
    // 医学图像：灰度读取 + 缩放 + /255 归一化（与训练侧 ToTensor+Resize(256,256) 一致）
    m_matInputImage = cv::imread(m_strImagePath, cv::IMREAD_GRAYSCALE);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated");
        return false;
    }

    m_cTimer.StartCpu();

    cv::Mat mat_resized;
    cv::resize(m_matInputImage, mat_resized, cv::Size(m_nvInputDims.d[3], m_nvInputDims.d[2]), 0, 0, cv::INTER_LINEAR);

    int i_area = m_nvInputDims.d[2] * m_nvInputDims.d[3];
    for (int i = 0; i < i_area; i++) {
        m_pInputMemory[0][i] = mat_resized.data[i] / 255.0f;
    }

    CUDA_CHECK(cudaMemcpyAsync(m_pInputMemory[1], m_pInputMemory[0], m_iInputSize,
                               cudaMemcpyKind::cudaMemcpyHostToDevice, m_cuStream));

    m_cTimer.StopCpu<Timer::ms>("preprocess(CPU)");
    return true;
}

bool UnetSegmenter::PreprocessGpu() {
    m_matInputImage = cv::imread(m_strImagePath, cv::IMREAD_GRAYSCALE);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated");
        return false;
    }

    m_cTimer.StartGpu();

    ResizeGrayGpu(m_matInputImage, m_pInputMemory[1], m_nvInputDims.d[2], m_nvInputDims.d[3],
                  InterpolationTactics::GPU_BILINEAR);

    m_cTimer.StopGpu("preprocess(GPU)");
    return true;
}

bool UnetSegmenter::PostprocessCpu() {
    m_cTimer.StartCpu();

    CUDA_CHECK(cudaMemcpyAsync(m_pOutputMemory[0], m_pOutputMemory[1], m_iOutputSize,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    int i_h = m_nvOutputDims.d[2];
    int i_w = m_nvOutputDims.d[3];

    // logits → 0/255 mask（阈值语义见头文件注释），并统计肝脏区域占比
    cv::Mat mat_prob(i_h, i_w, CV_32F, m_pOutputMemory[0]);
    cv::Mat mat_mask_small;
    cv::threshold(mat_prob, mat_mask_small, 0.5, 255, cv::THRESH_BINARY);
    mat_mask_small.convertTo(mat_mask_small, CV_8U);

    int   i_liver_pixels = cv::countNonZero(mat_mask_small);
    float f_ratio        = 100.0f * i_liver_pixels / (i_h * i_w);

    // mask 上采样回原图尺寸，红色半透明叠加
    cv::Mat mat_mask_full;
    if (m_matInputImage.data != nullptr && (m_matInputImage.cols != i_w || m_matInputImage.rows != i_h)) {
        cv::resize(mat_mask_small, mat_mask_full, m_matInputImage.size(), 0, 0, cv::INTER_LINEAR);
    }
    else {
        mat_mask_full = mat_mask_small;
    }
    m_matMask = mat_mask_full;

    cv::Mat mat_overlay;
    cv::cvtColor(m_matInputImage, mat_overlay, cv::COLOR_GRAY2BGR);
    cv::Mat mat_red(mat_overlay.size(), CV_8UC3, cv::Scalar(0, 0, 255));
    cv::Mat mat_blend;
    cv::addWeighted(mat_overlay, 0.65, mat_red, 0.35, 0, mat_blend);
    mat_blend.copyTo(mat_overlay, mat_mask_full);  // 仅在肝脏 mask 区域替换为红色半透明

    m_cTimer.StopCpu<Timer::ms>("postprocess(CPU)");

    if (!m_bSaveDebug) {
        return true;  // 体数据批量分割：调用方通过 GetMask() 取结果，跳过逐层落盘
    }

    std::string str_tag     = "liver-" + GetPrec(m_cParams.m_ePrec);
    m_strOutputPath         = ChangePath(m_strImagePath, "../result", ".png", str_tag);
    std::string str_out_dir = m_strOutputPath.substr(0, m_strOutputPath.find_last_of("/"));
    if (!IsFileExist(str_out_dir) && !CreateDir(str_out_dir)) {
        LOGE("Failed to create result directory: %s", str_out_dir.c_str());
    }

    LOG("\tResult:");
    LOG("\t\tLiver mask: %d pixels (%.2f%% of slice)", i_liver_pixels, f_ratio);
    cv::imwrite(m_strOutputPath, mat_overlay);

    // 单独保存二值 mask，便于后续评估
    std::string str_mask_path = m_strOutputPath;
    str_mask_path.replace(str_mask_path.find(".png"), 4, "-mask.png");
    cv::imwrite(str_mask_path, mat_mask_full);
    LOG("\tsave result to %s", m_strOutputPath.c_str());
    LOG("\tsave mask  to %s", str_mask_path.c_str());

    // 逐阶段耗时只在单图调试模式输出（Timer 内部用 LOGV，需日志级别 >= VERB，默认 sample 的 DEBUG 即可）；
    // 体数据批量分割（SetSaveDebug(false)）不打印，避免逐层刷屏
    m_cTimer.Show();

    return true;
}

std::unique_ptr<UnetSegmenter> MakeUnetSegmenter(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                                                 std::shared_ptr<InferBackend> pBackend) {
    return std::make_unique<UnetSegmenter>(strOnnxPath, cLevel, cParams, std::move(pBackend));
}
