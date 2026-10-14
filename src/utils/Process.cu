#include "include/utils/Process.h"
#include "include/utils/Logger.h"
#include "include/utils/CudaCheck.h"

TransInfo    cTransInfo;
AffineMatrix cAffineMatrix;

void WarpAffineInit(int iSrcH, int iSrcW, int iTarH, int iTarW) {
    cTransInfo.m_iSrcHeight    = iSrcH;
    cTransInfo.m_iSrcWidth     = iSrcW;
    cTransInfo.m_iTargetHeight = iTarH;
    cTransInfo.m_iTargetWidth  = iTarW;
    cAffineMatrix.Init(cTransInfo);
}

HOST_DEVICE void AffineTransformation(float arrTransMatrix[6], float fSrcX, float fSrcY, float* pTarX, float* pTarY) {
    *pTarX = arrTransMatrix[0] * fSrcX + arrTransMatrix[1] * fSrcY + arrTransMatrix[2];
    *pTarY = arrTransMatrix[3] * fSrcX + arrTransMatrix[4] * fSrcY + arrTransMatrix[5];
}

/// @brief 单通道双线性采样：4 邻域加权和 → round 取整 → /255 归一化
/// @param i_idx11..i_idx22 四个邻域像素在 pSrc 中的起始偏移，i_chan 为像素内的通道偏移
/// @param f_a1_1..f_a2_2 四个邻域对应的双线性权重
__device__ __forceinline__ float BilinearChannel(const uint8_t* pSrc, int i_idx11, int i_idx12, int i_idx21,
                                                 int i_idx22, int i_chan, float f_a1_1, float f_a1_2, float f_a2_1,
                                                 float f_a2_2) {
    float f_mix = f_a1_1 * pSrc[i_idx11 + i_chan] + f_a1_2 * pSrc[i_idx12 + i_chan] + f_a2_1 * pSrc[i_idx21 + i_chan] +
                  f_a2_2 * pSrc[i_idx22 + i_chan];
    return round(f_mix) / 255.0f;
}

/// @brief 一个像素的三通道双线性采样，按 RGB 顺序返回（源存 BGR，所以 x 取通道 2、z 取通道 0）
__device__ __forceinline__ float3 BilinearRgb(const uint8_t* pSrc, int i_idx11, int i_idx12, int i_idx21, int i_idx22,
                                              float f_a1_1, float f_a1_2, float f_a2_1, float f_a2_2) {
    float3 cRgb;
    cRgb.x = BilinearChannel(pSrc, i_idx11, i_idx12, i_idx21, i_idx22, 2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
    cRgb.y = BilinearChannel(pSrc, i_idx11, i_idx12, i_idx21, i_idx22, 1, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
    cRgb.z = BilinearChannel(pSrc, i_idx11, i_idx12, i_idx21, i_idx22, 0, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
    return cRgb;
}

__global__ void Nearest_BGR2RGB_NHWC2NCHW_Norm_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW,
                                                      int iSrcH, float fScaledW, float fScaledH, float* arrMean,
                                                      float* arrStd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 网格覆盖大于目标尺寸，越界线程直接返回，避免越界写
    }

    int i_src_y = floor((float)y * fScaledH);
    int i_src_x = floor((float)x * fScaledW);

    if (i_src_x >= 0 && i_src_y >= 0 && i_src_x < iSrcW && i_src_y < iSrcH) {
        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        int i_src_idx = (i_src_y * iSrcW + i_src_x) * 3;

        pTar[i_tar_idx + i_tar_area * 0] = (pSrc[i_src_idx + 2] / 255.0f - arrMean[2]) / arrStd[2];
        pTar[i_tar_idx + i_tar_area * 1] = (pSrc[i_src_idx + 1] / 255.0f - arrMean[1]) / arrStd[1];
        pTar[i_tar_idx + i_tar_area * 2] = (pSrc[i_src_idx + 0] / 255.0f - arrMean[0]) / arrStd[0];
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Norm_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW,
                                                       int iSrcH, float fScaledW, float fScaledH, float* arrMean,
                                                       float* arrStd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 >= 0 && i_src_x1 >= 0 && i_src_y1 < iSrcH && i_src_x1 < iSrcW) {
        // 邻域索引钳到边缘（等价 cv::resize 的 BORDER_REPLICATE），避免 x2 == iSrcW 时越界读
        i_src_y2 = min(i_src_y2, iSrcH - 1);
        i_src_x2 = min(i_src_x2, iSrcW - 1);

        float f_th = ((y + 0.5) * fScaledH - 0.5) - i_src_y1;
        float f_tw = ((x + 0.5) * fScaledW - 0.5) - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  //右下
        float f_a1_2 = f_tw * (1.0 - f_th);          //左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          //右上
        float f_a2_2 = f_tw * f_th;                  //左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  //左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  //右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  //左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  //右下

        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        float3 cRgb =
            BilinearRgb(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
        pTar[i_tar_idx + i_tar_area * 0] = (cRgb.x - arrMean[2]) / arrStd[2];
        pTar[i_tar_idx + i_tar_area * 1] = (cRgb.y - arrMean[1]) / arrStd[1];
        pTar[i_tar_idx + i_tar_area * 2] = (cRgb.z - arrMean[0]) / arrStd[0];
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Shift_Norm_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH,
                                                             int iSrcW, int iSrcH, float fScaledW, float fScaledH,
                                                             float* arrMean, float* arrStd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 >= 0 && i_src_x1 >= 0 && i_src_y1 < iSrcH && i_src_x1 < iSrcW) {
        // 邻域索引钳到边缘（等价 cv::resize 的 BORDER_REPLICATE），避免 x2 == iSrcW 时越界读
        i_src_y2 = min(i_src_y2, iSrcH - 1);
        i_src_x2 = min(i_src_x2, iSrcW - 1);

        float f_th = (float)y * fScaledH - i_src_y1;
        float f_tw = (float)x * fScaledW - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  // 右下
        float f_a1_2 = f_tw * (1.0 - f_th);          // 左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          // 右上
        float f_a2_2 = f_tw * f_th;                  // 左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  // 左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  // 右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  // 左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  // 右下

        y = y - int(iSrcH / (fScaledH * 2)) + int(iTarH / 2);
        x = x - int(iSrcW / (fScaledW * 2)) + int(iTarW / 2);

        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        float3 cRgb =
            BilinearRgb(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
        pTar[i_tar_idx + i_tar_area * 0] = (cRgb.x - arrMean[2]) / arrStd[2];
        pTar[i_tar_idx + i_tar_area * 1] = (cRgb.y - arrMean[1]) / arrStd[1];
        pTar[i_tar_idx + i_tar_area * 2] = (cRgb.z - arrMean[0]) / arrStd[0];
    }
}

__global__ void Nearest_BGR2RGB_NHWC2NCHW_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW, int iSrcH,
                                                 float fScaledW, float fScaledH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y = floor((float)y * fScaledH);
    int i_src_x = floor((float)x * fScaledW);

    if (i_src_x >= 0 && i_src_y >= 0 && i_src_x < iSrcW && i_src_y < iSrcH) {
        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        int i_src_idx = (i_src_y * iSrcW + i_src_x) * 3;

        pTar[i_tar_idx + i_tar_area * 0] = pSrc[i_src_idx + 2] / 255.0f;
        pTar[i_tar_idx + i_tar_area * 1] = pSrc[i_src_idx + 1] / 255.0f;
        pTar[i_tar_idx + i_tar_area * 2] = pSrc[i_src_idx + 0] / 255.0f;
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW,
                                                  int iSrcH, float fScaledW, float fScaledH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 >= 0 && i_src_x1 >= 0 && i_src_y1 < iSrcH && i_src_x1 < iSrcW) {
        // 邻域索引钳到边缘（等价 cv::resize 的 BORDER_REPLICATE），避免 x2 == iSrcW 时越界读
        i_src_y2 = min(i_src_y2, iSrcH - 1);
        i_src_x2 = min(i_src_x2, iSrcW - 1);

        float f_th = ((y + 0.5) * fScaledH - 0.5) - i_src_y1;
        float f_tw = ((x + 0.5) * fScaledW - 0.5) - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  //右下
        float f_a1_2 = f_tw * (1.0 - f_th);          //左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          //右上
        float f_a2_2 = f_tw * f_th;                  //左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  //左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  //右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  //左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  //右下

        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        float3 cRgb =
            BilinearRgb(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
        pTar[i_tar_idx + i_tar_area * 0] = cRgb.x;
        pTar[i_tar_idx + i_tar_area * 1] = cRgb.y;
        pTar[i_tar_idx + i_tar_area * 2] = cRgb.z;
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Shift_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW,
                                                        int iSrcH, float fScaledW, float fScaledH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 >= 0 && i_src_x1 >= 0 && i_src_y1 < iSrcH && i_src_x1 < iSrcW) {
        // 邻域索引钳到边缘（等价 cv::resize 的 BORDER_REPLICATE），避免 x2 == iSrcW 时越界读
        i_src_y2 = min(i_src_y2, iSrcH - 1);
        i_src_x2 = min(i_src_x2, iSrcW - 1);

        float f_th = (float)y * fScaledH - i_src_y1;
        float f_tw = (float)x * fScaledW - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  // 右下
        float f_a1_2 = f_tw * (1.0 - f_th);          // 左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          // 右上
        float f_a2_2 = f_tw * f_th;                  // 左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  // 左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  // 右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  // 左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  // 右下

        y = y - int(iSrcH / (fScaledH * 2)) + int(iTarH / 2);
        x = x - int(iSrcW / (fScaledW * 2)) + int(iTarW / 2);

        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        float3 cRgb =
            BilinearRgb(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
        pTar[i_tar_idx + i_tar_area * 0] = cRgb.x;
        pTar[i_tar_idx + i_tar_area * 1] = cRgb.y;
        pTar[i_tar_idx + i_tar_area * 2] = cRgb.z;
    }
}

__global__ void WarpAffine_BGR2RGB_Kernel(float* pTar, uint8_t* pSrc, TransInfo cTransInfo,
                                          AffineMatrix cAffineMatrix) {
    float i_src_x, i_src_y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cTransInfo.m_iTargetWidth || y >= cTransInfo.m_iTargetHeight) {
        return;  // 越界线程直接返回
    }

    AffineTransformation(cAffineMatrix.m_arrReverse, x + 0.5, y + 0.5, &i_src_x, &i_src_y);

    // OpenCV INTER_LINEAR 的基址约定：i_src_x/y 已是像元中心坐标，基址取 floor(s)、权重取 frac(s)；
    // 用 floor(s-0.5) 会在 frac<0.5 时取错一对邻居。
    // 落在 letterbox 黑边里的目标像素不写，保持调用方 memset 的 0，与 LetterboxCpu 的填充一致。
    if (x < cAffineMatrix.m_iPadX || x - cAffineMatrix.m_iPadX >= cAffineMatrix.m_iNewWidth ||
        y < cAffineMatrix.m_iPadY || y - cAffineMatrix.m_iPadY >= cAffineMatrix.m_iNewHeight) {
        return;
    }

    // 先钳源坐标、再算基址与权重，等价 cv::resize 的 BORDER_REPLICATE：
    // 边缘行/列的源坐标会略微越出 [0, src-1]（如 -0.109），cv::resize 是 snap 到边缘像素，
    // 如果只钳索引而保留 frac，会把这半格插到图像内侧的相邻像素上，整行/列偏一个像素。
    float f_src_x = min(max(i_src_x, 0.0f), (float)(cTransInfo.m_iSrcWidth - 1));
    float f_src_y = min(max(i_src_y, 0.0f), (float)(cTransInfo.m_iSrcHeight - 1));

    int i_src_x1 = (int)floor(f_src_x);
    int i_src_y1 = (int)floor(f_src_y);
    int i_src_x2 = min(i_src_x1 + 1, cTransInfo.m_iSrcWidth - 1);
    int i_src_y2 = min(i_src_y1 + 1, cTransInfo.m_iSrcHeight - 1);

    float f_tw = f_src_x - i_src_x1;
    float f_th = f_src_y - i_src_y1;

    float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);
    float f_a1_2 = f_tw * (1.0 - f_th);
    float f_a2_1 = (1.0 - f_tw) * f_th;
    float f_a2_2 = f_tw * f_th;

    int i_src_idx1_1 = (i_src_y1 * cTransInfo.m_iSrcWidth + i_src_x1) * 3;
    int i_src_idx1_2 = (i_src_y1 * cTransInfo.m_iSrcWidth + i_src_x2) * 3;
    int i_src_idx2_1 = (i_src_y2 * cTransInfo.m_iSrcWidth + i_src_x1) * 3;
    int i_src_idx2_2 = (i_src_y2 * cTransInfo.m_iSrcWidth + i_src_x2) * 3;

    int i_tar_idx  = y * cTransInfo.m_iTargetWidth + x;
    int i_tar_area = cTransInfo.m_iTargetWidth * cTransInfo.m_iTargetHeight;

    float3 cRgb =
        BilinearRgb(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, f_a1_1, f_a1_2, f_a2_1, f_a2_2);
    pTar[i_tar_idx + i_tar_area * 0] = cRgb.x;
    pTar[i_tar_idx + i_tar_area * 1] = cRgb.y;
    pTar[i_tar_idx + i_tar_area * 2] = cRgb.z;
}

__global__ void BilinearGray_Norm_Kernel(float* pTar, uint8_t* pSrc, int iTarW, int iTarH, int iSrcW, int iSrcH,
                                         float fScaledW, float fScaledH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= iTarW || y >= iTarH) {
        return;  // 越界线程直接返回
    }

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 >= iSrcH || i_src_x2 >= iSrcW) {
        // 越界像素置零（letterbox 边界）
        pTar[y * iTarW + x] = 0.0f;
        return;
    }

    float f_th = ((y + 0.5) * fScaledH - 0.5) - i_src_y1;
    float f_tw = ((x + 0.5) * fScaledW - 0.5) - i_src_x1;

    float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  //右下
    float f_a1_2 = f_tw * (1.0 - f_th);          //左下
    float f_a2_1 = (1.0 - f_tw) * f_th;          //右上
    float f_a2_2 = f_tw * f_th;                  //左上

    int i_src_idx1_1 = i_src_y1 * iSrcW + i_src_x1;
    int i_src_idx1_2 = i_src_y1 * iSrcW + i_src_x2;
    int i_src_idx2_1 = i_src_y2 * iSrcW + i_src_x1;
    int i_src_idx2_2 = i_src_y2 * iSrcW + i_src_x2;

    // 灰度图每像素只占 1 字节，故通道偏移传 0，上面四个索引已经是像素下标
    pTar[y * iTarW + x] = BilinearChannel(pSrc, i_src_idx1_1, i_src_idx1_2, i_src_idx2_1, i_src_idx2_2, 0, f_a1_1,
                                          f_a1_2, f_a2_1, f_a2_2);
}

void ResizeGrayGpu(cv::Mat& matHostSrc, float* pDeviceTar, const int& iTarH, const int& iTarW,
                   InterpolationTactics eInterpolation) {
    // 单通道灰度图双线性缩放 + /255 归一化（医学图像分割预处理，如 UNet 肝脏分割）
    if (eInterpolation != InterpolationTactics::GPU_BILINEAR) {
        LOGE("ERROR: ResizeGrayGpu only supports GPU_BILINEAR tactic. Program terminated");
        exit(1);
    }

    uint8_t* d_src    = nullptr;
    int      i_height = matHostSrc.rows;
    int      i_width  = matHostSrc.cols;

    int src_size = i_height * i_width * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc(&d_src, src_size));
    CUDA_CHECK(cudaMemset(pDeviceTar, 0, iTarH * iTarW * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, matHostSrc.data, src_size, cudaMemcpyHostToDevice));

    dim3  dim_block(32, 32, 1);
    dim3  dim_grid(iTarW / 32 + 1, iTarH / 32 + 1, 1);
    float f_scaled_h = (float)i_height / iTarH;
    float f_scaled_w = (float)i_width / iTarW;

    BilinearGray_Norm_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, d_src, iTarW, iTarH, i_width, i_height, f_scaled_w,
                                                      f_scaled_h);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_src));
}

void ResizeBilinearGpu(float* pDeviceTar, uint8_t* pDeviceSrc, int iTarW, int iTarH, int iSrcW, int iSrcH,
                       float* arrMean, float* arrStd, InterpolationTactics eTac) {
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(iTarW / 32 + 1, iTarH / 32 + 1, 1);

    float f_scaled_h = (float)iSrcH / iTarH;
    float f_scaled_w = (float)iSrcW / iTarW;
    float f_scale    = (f_scaled_h > f_scaled_w ? f_scaled_h : f_scaled_w);

    switch (eTac) {
        case InterpolationTactics::GPU_NEAREST:
            Nearest_BGR2RGB_NHWC2NCHW_Norm_Kernel<<<dim_grid, dim_block>>>(
                pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, f_scaled_w, f_scaled_h, arrMean, arrStd);
            break;
        case InterpolationTactics::GPU_NEAREST_CENTER:
            Nearest_BGR2RGB_NHWC2NCHW_Norm_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW,
                                                                           iSrcH, f_scale, f_scale, arrMean, arrStd);
            break;
        case InterpolationTactics::GPU_BILINEAR:
            Bilinear_BGR2RGB_NHWC2NCHW_Norm_Kernel<<<dim_grid, dim_block>>>(
                pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, f_scaled_w, f_scaled_h, arrMean, arrStd);
            break;
        case InterpolationTactics::GPU_BILINEAR_CENTER:
            Bilinear_BGR2RGB_NHWC2NCHW_Shift_Norm_Kernel<<<dim_grid, dim_block>>>(
                pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, f_scale, f_scale, arrMean, arrStd);
            break;
        case InterpolationTactics::GPU_WARP_AFFINE:
            // warpaffine 核函数自带 /255 归一化，不使用 mean/std
            WarpAffineInit(iSrcH, iSrcW, iTarH, iTarW);
            WarpAffine_BGR2RGB_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, cTransInfo, cAffineMatrix);
            break;
        default:
            LOGE("ERROR: Wrong GPU resize InterpolationTactics selected. Program terminated");
            exit(1);
    }
}

void ResizeBilinearGpu(float* pDeviceTar, uint8_t* pDeviceSrc, int iTarW, int iTarH, int iSrcW, int iSrcH,
                       InterpolationTactics eTac) {
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(iTarW / 32 + 1, iTarH / 32 + 1, 1);

    float f_scaled_h = (float)iSrcH / iTarH;
    float f_scaled_w = (float)iSrcW / iTarW;
    float f_scale    = (f_scaled_h > f_scaled_w ? f_scaled_h : f_scaled_w);

    switch (eTac) {
        case InterpolationTactics::GPU_NEAREST:
            Nearest_BGR2RGB_NHWC2NCHW_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW,
                                                                      iSrcH, f_scaled_w, f_scaled_h);
            break;
        case InterpolationTactics::GPU_NEAREST_CENTER:
            Nearest_BGR2RGB_NHWC2NCHW_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW,
                                                                      iSrcH, f_scale, f_scale);
            break;
        case InterpolationTactics::GPU_BILINEAR:
            Bilinear_BGR2RGB_NHWC2NCHW_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW,
                                                                       iSrcH, f_scaled_w, f_scaled_h);
            break;
        case InterpolationTactics::GPU_BILINEAR_CENTER:
            Bilinear_BGR2RGB_NHWC2NCHW_Shift_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, iTarW, iTarH,
                                                                             iSrcW, iSrcH, f_scale, f_scale);
            break;
        case InterpolationTactics::GPU_WARP_AFFINE:
            WarpAffineInit(iSrcH, iSrcW, iTarH, iTarW);
            WarpAffine_BGR2RGB_Kernel<<<dim_grid, dim_block>>>(pDeviceTar, pDeviceSrc, cTransInfo, cAffineMatrix);
            break;
        default:
            LOGE("ERROR: Wrong GPU resize InterpolationTactics selected. Program terminated");
            exit(1);
    }
}