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
    TransInfo()         = default;
    TransInfo(int iSrcW, int iSrcH, int iTarW, int iTarH)
        : m_iSrcWidth(iSrcW)
        , m_iSrcHeight(iSrcH)
        , m_iTargetWidth(iTarW)
        , m_iTargetHeight(iTarH) {
    }
};

struct AffineMatrix {
    float m_arrForward[6];
    float m_arrReverse[6];     ///< 供核函数采样：入参已是像素中心 +0.5，故含 -0.5 半像素项
    float m_arrReverseBox[6];  ///< 供检测框映回原图：纯坐标变换，不含半像素项
    float m_fForwardScale;
    float m_fReverseScale;
    int   m_iNewWidth  = 0;  ///< 截断后的缩放尺寸，与 LetterboxCpu 的 int(srcW*scale) 完全一致
    int   m_iNewHeight = 0;
    int   m_iPadX      = 0;  ///< 整数居中偏移，与 LetterboxCpu 的 (tarW-newW)/2 一致
    int   m_iPadY      = 0;

    void CalcForwardMatrix(TransInfo cTrans) {
        m_arrForward[0] = m_fForwardScale;
        m_arrForward[1] = 0;
        m_arrForward[2] = -m_fForwardScale * cTrans.m_iSrcWidth * 0.5 + cTrans.m_iTargetWidth * 0.5;
        m_arrForward[3] = 0;
        m_arrForward[4] = m_fForwardScale;
        m_arrForward[5] = -m_fForwardScale * cTrans.m_iSrcHeight * 0.5 + cTrans.m_iTargetHeight * 0.5;
    };

    /// @brief 反向矩阵必须跟 cv::resize + 整数居中保持一致：
    ///        ① 实际缩放是 newW/srcW（各轴独立，newW 由截断得到，可能异于各向同性的 forwardScale）；
    ///        ② padding 为奇数时多出的 1 像素在右下，不是精确居中；
    ///        否则 GPU 会整体偏半个像素，表现为全图上下/左右行 50/50 混合（模糊）。
    void CalcReverseMatrix(TransInfo cTrans) {
        float f_rev_x = (m_iNewWidth > 0) ? (float)cTrans.m_iSrcWidth / m_iNewWidth : m_fReverseScale;
        float f_rev_y = (m_iNewHeight > 0) ? (float)cTrans.m_iSrcHeight / m_iNewHeight : m_fReverseScale;

        m_arrReverse[0] = f_rev_x;
        m_arrReverse[1] = 0;
        m_arrReverse[2] = -f_rev_x * m_iPadX - 0.5f;
        m_arrReverse[3] = 0;
        m_arrReverse[4] = f_rev_y;
        m_arrReverse[5] = -f_rev_y * m_iPadY - 0.5f;

        m_arrReverseBox[0] = f_rev_x;
        m_arrReverseBox[1] = 0;
        m_arrReverseBox[2] = -f_rev_x * m_iPadX;
        m_arrReverseBox[3] = 0;
        m_arrReverseBox[4] = f_rev_y;
        m_arrReverseBox[5] = -f_rev_y * m_iPadY;
    };

    void Init(TransInfo cTrans) {
        float f_scaled_w = (float)cTrans.m_iTargetWidth / cTrans.m_iSrcWidth;
        float f_scaled_h = (float)cTrans.m_iTargetHeight / cTrans.m_iSrcHeight;
        m_fForwardScale  = (f_scaled_w < f_scaled_h ? f_scaled_w : f_scaled_h);
        m_fReverseScale  = 1 / m_fForwardScale;

        // 与 LetterboxCpu 用完全相同的表达式（float 乘法后截断），保证两边的缩放尺寸相同
        m_iNewWidth  = (int)(cTrans.m_iSrcWidth * m_fForwardScale);
        m_iNewHeight = (int)(cTrans.m_iSrcHeight * m_fForwardScale);
        m_iPadX      = (m_iNewWidth < cTrans.m_iTargetWidth) ? (cTrans.m_iTargetWidth - m_iNewWidth) / 2 : 0;
        m_iPadY      = (m_iNewHeight < cTrans.m_iTargetHeight) ? (cTrans.m_iTargetHeight - m_iNewHeight) / 2 : 0;

        CalcForwardMatrix(cTrans);
        CalcReverseMatrix(cTrans);
    }
};

extern TransInfo    cTrans;
extern AffineMatrix cAffineMatrix;

void WarpAffineInit(int iSrcH, int iSrcW, int iTarH, int iTarW);

cv::Mat ResizeCpu(cv::Mat& matSrc, const int& iTarH, const int& iTarW, float* arrMean, float* arrStd,
                  InterpolationTactics eInterpolation);

cv::Mat ResizeCpu(cv::Mat& matSrc, const int& iTarH, const int& iTarW, InterpolationTactics eInterpolation);

void ResizeGpu(cv::Mat& matSrc, float* pTar, const int& iTarH, const int& iTarW, float* arrMean, float* arrStd,
               InterpolationTactics eInterpolation);

void ResizeGpu(cv::Mat& matSrc, float* pTar, const int& iTarH, const int& iTarW, InterpolationTactics eInterpolation);

/// @brief 单通道灰度图 GPU 双线性缩放 + /255 归一化（医学图像分割预处理）
void ResizeGrayGpu(cv::Mat& matSrc, float* pTar, const int& iTarH, const int& iTarW,
                   InterpolationTactics eInterpolation);

void ResizeBilinearGpu(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcH, int iSrcW, float* arrMean,
                       float* arrStd, InterpolationTactics eInterpolation);

void ResizeBilinearGpu(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcH, int iSrcW,
                       InterpolationTactics eInterpolation);

HOST_DEVICE void AffineTransformation(float* trans_matrix, float fSrcX, float fSrcY, float* pTarX, float* pTarY);

/// @brief CPU letterbox 预处理：等比缩放 + 黑边填充 + NHWC→NCHW + /255 归一化
/// @param matSrc 输入图像 (BGR, uint8)
/// @param pDst 输出缓冲区 (float, NCHW, 已归一化)
/// @param iTarH 目标高度
/// @param iTarW 目标宽度
/// @return 填充后的完整图像 (用于调试/可视化)
cv::Mat LetterboxCpu(const cv::Mat& matSrc, float* pDst, int iTarH, int iTarW);
