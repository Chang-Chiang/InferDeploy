#include "include/utils/Process.h"
#include "include/utils/Logger.h"
#include "include/utils/CudaCheck.h"

cv::Mat ResizeCpu(
    cv::Mat &matSrc, 
    const int &iTarH, const int &iTarW, 
    float* arrMean, float* arrStd,
    InterpolationTactics eInterpolation) 
{
    cv::Mat mat_tar;

    int i_height   = matSrc.rows;
    int i_width    = matSrc.cols;
    float f_dim    = std::max(i_height, i_width);
    int i_resize_h = ((i_height / f_dim) * iTarH);
    int i_resize_w = ((i_width / f_dim) * iTarW);
    int i_x_offset = (iTarW - i_resize_w) / 2;
    int i_y_offset = (iTarH - i_resize_h) / 2;
    i_resize_w     = iTarW;
    i_resize_h     = iTarH;

    // BGR2RGB
    cv::cvtColor(matSrc, matSrc, cv::COLOR_BGR2RGB);

    // Resize
    switch (eInterpolation) {
        case InterpolationTactics::CPU_NEAREST: 
            cv::resize(matSrc, mat_tar, cv::Size(i_resize_w, i_resize_h), 0, 0, cv::INTER_LINEAR);
            break;
        case InterpolationTactics::CPU_BILINEAR: 
            cv::resize(matSrc, mat_tar, cv::Size(i_resize_w, i_resize_h), 0, 0, cv::INTER_NEAREST);
            break;
        default:
            LOGE("ERROR: Wrong CPU resize tactics selected. Program terminated");
            exit(1);
    }
    return mat_tar;
}

void ResizeGpu(
    cv::Mat &matHostSrc, float* pDeviceTar, 
    const int& iTarH, const int& iTarW, 
    float* arrMean, float* arrStd, 
    InterpolationTactics eInterpolation) 
{
    float*   arr_mean = nullptr;
    float*   arr_std  = nullptr;
    uint8_t* d_src    = nullptr;

    int i_height = matHostSrc.rows;
    int i_width  = matHostSrc.cols;
    int chan     = 3;

    int src_size  = i_height * i_width * chan * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);

    // 分配device上的src和mean, std的内存
    CUDA_CHECK(cudaMalloc((void**)&d_src, src_size));
    CUDA_CHECK(cudaMalloc((void**)&arr_mean, norm_size));
    CUDA_CHECK(cudaMalloc((void**)&arr_std, norm_size));

    // letterbox 边界像素不会被核函数写入，先清零目标缓冲，避免未初始化内存污染输入
    CUDA_CHECK(cudaMemset(pDeviceTar, 0, iTarH * iTarW * 3 * sizeof(float)));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, matHostSrc.data, src_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(arr_mean, arrMean, norm_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(arr_std, arrStd, norm_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    ResizeBilinearGpu(pDeviceTar, d_src, iTarW, iTarH, i_width, i_height, arr_mean, arr_std, eInterpolation);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(arr_std));
    CUDA_CHECK(cudaFree(arr_mean));
    CUDA_CHECK(cudaFree(d_src));

    // 因为接下来会继续在gpu上进行处理，所以这里不用把结果返回到host
}

// 根据比例进行缩放 (GPU版本)
void ResizeGpu(
    cv::Mat &matHostSrc, float* pDeviceTar, 
    const int& iTarH, const int& iTarW, 
    InterpolationTactics eInterpolation) 
{
    uint8_t* d_src  = nullptr;

    int i_height   = matHostSrc.rows;
    int i_width    = matHostSrc.cols;
    int chan     = 3;

    int src_size  = i_height * i_width * chan * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);


    // 分配device上的src的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));

    // letterbox 边界像素不会被核函数写入，先清零目标缓冲，避免未初始化内存污染输入
    CUDA_CHECK(cudaMemset(pDeviceTar, 0, iTarH * iTarW * 3 * sizeof(float)));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, matHostSrc.data, src_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    ResizeBilinearGpu(pDeviceTar, d_src, iTarW, iTarH, i_width, i_height, nullptr, nullptr, eInterpolation);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_src));

    // 因为接下来会继续在gpu上进行处理，所以这里不用把结果返回到host
}

cv::Mat LetterboxCpu(const cv::Mat& matSrc, float* pDst, int iTarH, int iTarW) {
    int   i_input_w = matSrc.cols;
    int   i_input_h = matSrc.rows;
    float f_scale   = std::min(float(iTarW) / i_input_w, float(iTarH) / i_input_h);
    int   i_new_w   = int(i_input_w * f_scale);
    int   i_new_h   = int(i_input_h * f_scale);

    cv::Mat mat_resized;
    cv::resize(matSrc, mat_resized, cv::Size(i_new_w, i_new_h));

    // 创建目标画布并居中放置
    cv::Mat mat_tar(iTarH, iTarW, CV_8UC3, cv::Scalar(0, 0, 0));
    int i_x = (i_new_w < iTarW) ? (iTarW - i_new_w) / 2 : 0;
    int i_y = (i_new_h < iTarH) ? (iTarH - i_new_h) / 2 : 0;
    cv::Mat mat_roi = mat_tar(cv::Rect(i_x, i_y, i_new_w, i_new_h));
    mat_resized.copyTo(mat_roi);

    // NHWC → NCHW, /255 归一化
    int i_img_area = iTarH * iTarW;
    int i_offset_ch0 = i_img_area * 0;
    int i_offset_ch1 = i_img_area * 1;
    int i_offset_ch2 = i_img_area * 2;
    int i_channel = 3;

    for (int i = 0; i < iTarH; i++) {
        for (int j = 0; j < iTarW; j++) {
            int i_idx = i * iTarW * i_channel + j * i_channel;
            pDst[i_offset_ch2++] = mat_tar.data[i_idx + 0] / 255.0f;
            pDst[i_offset_ch1++] = mat_tar.data[i_idx + 1] / 255.0f;
            pDst[i_offset_ch0++] = mat_tar.data[i_idx + 2] / 255.0f;
        }
    }

    return mat_tar;
}