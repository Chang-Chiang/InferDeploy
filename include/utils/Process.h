#pragma once

#include "opencv2/opencv.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

enum class InterpolationTactics : int32_t {
    CPU_NEAREST         = 0,
    CPU_BILINEAR        = 1,
    GPU_NEAREST         = 2,
    GPU_NEAREST_CENTER  = 3,
    GPU_BILINEAR        = 4,
    GPU_BILINEAR_CENTER = 5,
    GPU_WARP_AFFINE     = 6,
};

struct TransInfo {
    int m_iSrcWidth     = 0;
    int m_iSrcHeight    = 0;
    int m_iTargetWidth  = 0;
    int m_iTargetHeight = 0;
    TransInfo() = default;
    TransInfo(int iSrcW, int iSrcH, int iTarW, int iTarH)
        : m_iSrcWidth(iSrcW)
        , m_iSrcHeight(iSrcH)
        , m_iTargetWidth(iTarW)
        , m_iTargetHeight(iTarH) { }
};

struct AffineMatrix {
    float m_arrForward[6];
    float m_arrReverse[6];
    float m_fForwardScale;
    float m_fReverseScale;

    void CalcForwardMatrix(TransInfo cTrans) {
        m_arrForward[0] = m_fForwardScale;
        m_arrForward[1] = 0;
        m_arrForward[2] = - m_fForwardScale * cTrans.m_iSrcWidth * 0.5 + cTrans.m_iTargetWidth * 0.5;
        m_arrForward[3] = 0;
        m_arrForward[4] = m_fForwardScale;
        m_arrForward[5] = - m_fForwardScale * cTrans.m_iSrcHeight * 0.5 + cTrans.m_iTargetHeight * 0.5;
    };

    void CalcReverseMatrix(TransInfo cTrans) {
        m_arrReverse[0] = m_fReverseScale;
        m_arrReverse[1] = 0;
        m_arrReverse[2] = - m_fReverseScale * cTrans.m_iTargetWidth * 0.5 + cTrans.m_iSrcWidth * 0.5;
        m_arrReverse[3] = 0;
        m_arrReverse[4] = m_fReverseScale;
        m_arrReverse[5] = - m_fReverseScale * cTrans.m_iTargetHeight * 0.5 + cTrans.m_iSrcHeight * 0.5;
    };

    void Init(TransInfo cTrans) {
        float f_scaled_w = (float)cTrans.m_iTargetWidth / cTrans.m_iSrcWidth;
        float f_scaled_h = (float)cTrans.m_iTargetHeight / cTrans.m_iSrcHeight;
        m_fForwardScale = (f_scaled_w < f_scaled_h ? f_scaled_w : f_scaled_h);
        m_fReverseScale = 1 / m_fForwardScale;
    
        CalcForwardMatrix(cTrans);
        CalcReverseMatrix(cTrans);
    }
};

extern  TransInfo    cTrans;
extern  AffineMatrix cAffineMatrix;

void    WarpAffineInit(int iSrcH, int iSrcW, int iTarH, int iTarW);

cv::Mat ResizeCpu(
    cv::Mat &matSrc, 
    const int& iTarH, const int& iTarW, 
    float* arrMean, float* arrStd, 
    InterpolationTactics eInterpolation);

cv::Mat ResizeCpu(
    cv::Mat &matSrc, 
    const int& iTarH, const int& iTarW, 
    InterpolationTactics eInterpolation);

void ResizeGpu(
    cv::Mat &matSrc, 
    float* pTar, 
    const int& iTarH, const int& iTarW, 
    float* arrMean, float* arrStd, 
    InterpolationTactics eInterpolation);

void ResizeGpu(
    cv::Mat &matSrc, float* pTar, 
    const int& iTarH, const int& iTarW, 
    InterpolationTactics eInterpolation);

/// @brief 单通道灰度图 GPU 双线性缩放 + /255 归一化（医学图像分割预处理）
void ResizeGrayGpu(
    cv::Mat &matSrc, float* pTar, 
    const int& iTarH, const int& iTarW, 
    InterpolationTactics eInterpolation);

void ResizeBilinearGpu(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcH, int iSrcW, 
    float* arrMean, float* arrStd, 
    InterpolationTactics eInterpolation);

void ResizeBilinearGpu(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcH, int iSrcW, 
    InterpolationTactics eInterpolation);

HOST_DEVICE void AffineTransformation(float* trans_matrix, int src_x, int src_y, float* tar_x, float* tar_y);

/// @brief CPU letterbox 预处理：等比缩放 + 黑边填充 + NHWC→NCHW + /255 归一化
/// @param matSrc 输入图像 (BGR, uint8)
/// @param pDst 输出缓冲区 (float, NCHW, 已归一化)
/// @param iTarH 目标高度
/// @param iTarW 目标宽度
/// @return 填充后的完整图像 (用于调试/可视化)
cv::Mat LetterboxCpu(const cv::Mat& matSrc, float* pDst, int iTarH, int iTarW);
