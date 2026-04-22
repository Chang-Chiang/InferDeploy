#include "include/utils/Process.h"

TransInfo    cTransInfo;
AffineMatrix cAffineMatrix;

void WarpAffineInit(int iSrcH, int iSrcW, int iTarH, int iTarW) {
    cTransInfo.src_h = iSrcH;
    cTransInfo.src_w = iSrcW;
    cTransInfo.tar_h = iTarH;
    cTransInfo.tar_w = iTarW;
    cAffineMatrix.Init(cTransInfo);
}

__host__ __device__ void AffineTransformation(
    float arrTransMatrix[6], 
    int iSrcX, int iSrcY, 
    float* pTarX, float* pTarY)
{
    *pTarX = arrTransMatrix[0] * iSrcX + arrTransMatrix[1] * iSrcY + arrTransMatrix[2];
    *pTarY = arrTransMatrix[3] * iSrcX + arrTransMatrix[4] * iSrcY + arrTransMatrix[5];
}

__global__ void Nearest_BGR2RGB_NHWC2NCHW_Norm_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH,
    float fScaledW, float fScaledH,
    float* arrMean, float* arrStd) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y = floor((float)y * fScaledH);
    int i_src_x = floor((float)x * fScaledW);

    if (i_src_x < 0 || i_src_y < 0 || i_src_x > iSrcW || i_src_y > iSrcH) {

    }
    else {
        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        int i_src_idx = (i_src_y * iSrcW + i_src_x) * 3;

        pTar[i_tar_idx + i_tar_area * 0] = (pSrc[i_src_idx + 2] / 255.0f - arrMean[2]) / arrStd[2];
        pTar[i_tar_idx + i_tar_area * 1] = (pSrc[i_src_idx + 1] / 255.0f - arrMean[1]) / arrStd[1];
        pTar[i_tar_idx + i_tar_area * 2] = (pSrc[i_src_idx + 0] / 255.0f - arrMean[0]) / arrStd[0];
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Norm_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    float fScaledW, float fScaledH,
    float* arrMean, float* arrStd) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 > iSrcH || i_src_x2 > iSrcW) {

    }
    else {
        float f_th   = ((y + 0.5) * fScaledH - 0.5) - i_src_y1;
        float f_tw   = ((x + 0.5) * fScaledW - 0.5) - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  //右下
        float f_a1_2 = f_tw * (1.0 - f_th);          //左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          //右上
        float f_a2_2 = f_tw * f_th;                  //左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  //左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  //右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  //左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  //右下

        int i_tar_idx    = y * iTarW  + x;
        int i_tar_area   = iTarW * iTarH;

        pTar[i_tar_idx + i_tar_area * 0] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 2] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 2] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 2] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 2])) / 255.0f - arrMean[2]) / arrStd[2];

        pTar[i_tar_idx + i_tar_area * 1] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 1] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 1] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 1] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 1])) / 255.0f - arrMean[1]) / arrStd[1];

        pTar[i_tar_idx + i_tar_area * 2] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 0] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 0] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 0] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 0])) / 255.0f - arrMean[0]) / arrStd[0];

    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Shift_Norm_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    float fScaledW, float fScaledH,
    float* arrMean, float* arrStd) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 > iSrcH || i_src_x2 > iSrcW) {

    }
    else {
        float f_th   = (float)y * fScaledH - i_src_y1;
        float f_tw   = (float)x * fScaledW - i_src_x1;

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

        int i_tar_idx    = (y * iTarW  + x) * 3;
        int i_tar_area   = iTarW * iTarH;

        pTar[i_tar_idx + i_tar_area * 0] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 2] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 2] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 2] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 2])) / 255.0f - arrMean[2]) / arrStd[2];

        pTar[i_tar_idx + i_tar_area * 1] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 1] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 1] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 1] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 1])) / 255.0f - arrMean[1]) / arrStd[1];

        pTar[i_tar_idx + i_tar_area * 2] = 
            (round((f_a1_1 * pSrc[i_src_idx1_1 + 0] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 0] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 0] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 0])) / 255.0f - arrMean[0]) / arrStd[0];
    }
}


__global__ void Nearest_BGR2RGB_NHWC2NCHW_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH,
    float fScaledW, float fScaledH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y = floor((float)y * fScaledH);
    int i_src_x = floor((float)x * fScaledW);

    if (i_src_x < 0 || i_src_y < 0 || i_src_x > iSrcW || i_src_y > iSrcH) {

    }
    else {
        int i_tar_idx  = y * iTarW + x;
        int i_tar_area = iTarW * iTarH;

        int i_src_idx = (i_src_y * iSrcW + i_src_x) * 3;

        pTar[i_tar_idx + i_tar_area * 0] = pSrc[i_src_idx + 2] / 255.0f;
        pTar[i_tar_idx + i_tar_area * 1] = pSrc[i_src_idx + 1] / 255.0f;
        pTar[i_tar_idx + i_tar_area * 2] = pSrc[i_src_idx + 0] / 255.0f;
    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    float fScaledW, float fScaledH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 > iSrcH || i_src_x2 > iSrcW) {

    }
    else {
        float f_th   = ((y + 0.5) * fScaledH - 0.5) - i_src_y1;
        float f_tw   = ((x + 0.5) * fScaledW - 0.5) - i_src_x1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);  //右下
        float f_a1_2 = f_tw * (1.0 - f_th);          //左下
        float f_a2_1 = (1.0 - f_tw) * f_th;          //右上
        float f_a2_2 = f_tw * f_th;                  //左上

        int i_src_idx1_1 = (i_src_y1 * iSrcW + i_src_x1) * 3;  //左上
        int i_src_idx1_2 = (i_src_y1 * iSrcW + i_src_x2) * 3;  //右上
        int i_src_idx2_1 = (i_src_y2 * iSrcW + i_src_x1) * 3;  //左下
        int i_src_idx2_2 = (i_src_y2 * iSrcW + i_src_x2) * 3;  //右下

        int i_tar_idx    = y * iTarW  + x;
        int i_tar_area   = iTarW * iTarH;

        pTar[i_tar_idx + i_tar_area * 0] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 2] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 2] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 2] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 2])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 1] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 1] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 1] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 1] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 1])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 2] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 0] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 0] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 0] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 0])) / 255.0f;

    }
}

__global__ void Bilinear_BGR2RGB_NHWC2NCHW_Shift_Norm_Kernel(
    float* pTar, uint8_t* pSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    float fScaledW, float fScaledH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_src_y1 = floor((y + 0.5) * fScaledH - 0.5);
    int i_src_x1 = floor((x + 0.5) * fScaledW - 0.5);
    int i_src_y2 = i_src_y1 + 1;
    int i_src_x2 = i_src_x1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 > iSrcH || i_src_x2 > iSrcW) {

    }
    else {
        float f_th   = (float)y * fScaledH - i_src_y1;
        float f_tw   = (float)x * fScaledW - i_src_x1;

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

        int i_tar_idx    = y * iTarW  + x;
        int i_tar_area   = iTarW * iTarH;

        pTar[i_tar_idx + i_tar_area * 0] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 2] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 2] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 2] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 2])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 1] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 1] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 1] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 1] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 1])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 2] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 0] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 0] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 0] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 0])) / 255.0f;
    }
}

__global__ void WarpAffine_BGR2RGB_Kernel(
    float* pTar, uint8_t* pSrc, 
    TransInfo cTransInfo,
    AffineMatrix cAffineMatrix)
{
    float i_src_x, i_src_y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    affine_transformation(cAffineMatrix.reverse, x + 0.5, y + 0.5, &i_src_x, &i_src_y);

    int i_src_x1 = floor(i_src_x - 0.5);
    int i_src_y1 = floor(i_src_y - 0.5);
    int i_src_x2 = i_src_x1 + 1;
    int i_src_y2 = i_src_y1 + 1;

    if (i_src_y1 < 0 || i_src_x1 < 0 || i_src_y2 > cTransInfo.src_h || i_src_x2 > cTransInfo.src_w) {

    }
    else {
        float f_tw   = i_src_x - i_src_x1;
        float f_th   = i_src_y - i_src_y1;

        float f_a1_1 = (1.0 - f_tw) * (1.0 - f_th);
        float f_a1_2 = f_tw * (1.0 - f_th);
        float f_a2_1 = (1.0 - f_tw) * f_th;
        float f_a2_2 = f_tw * f_th;

        int i_src_idx1_1 = (i_src_y1 * cTransInfo.src_w + i_src_x1) * 3;
        int i_src_idx1_2 = (i_src_y1 * cTransInfo.src_w + i_src_x2) * 3;
        int i_src_idx2_1 = (i_src_y2 * cTransInfo.src_w + i_src_x1) * 3;
        int i_src_idx2_2 = (i_src_y2 * cTransInfo.src_w + i_src_x2) * 3;

        int i_tar_idx    = y * cTransInfo.tar_w  + x;
        int i_tar_area   = cTransInfo.tar_w * cTransInfo.tar_h;

        pTar[i_tar_idx + i_tar_area * 0] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 2] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 2] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 2] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 2])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 1] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 1] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 1] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 1] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 1])) / 255.0f;

        pTar[i_tar_idx + i_tar_area * 2] = 
            round((f_a1_1 * pSrc[i_src_idx1_1 + 0] + 
                   f_a1_2 * pSrc[i_src_idx1_2 + 0] +
                   f_a2_1 * pSrc[i_src_idx2_1 + 0] +
                   f_a2_2 * pSrc[i_src_idx2_2 + 0])) / 255.0f;
    }
}

void ResizeBilinearGpu(
    float* pDeviceTar, uint8_t* pDeviceSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    float* arrMean, float* arrStd,
    InterpolationTactics eTac) 
{
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(iTarW / 32 + 1, iTarH / 32 + 1, 1);

    float fScaledH = (float)iSrcH / iTarH;
    float fScaledW = (float)iSrcW / iTarW;
    float scale = (fScaledH > fScaledW ? fScaledH : fScaledW);

    switch (eTac) {
    case InterpolationTactics::GPU_NEAREST:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dim_grid, dim_block>>>
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, fScaledW, fScaledH, arrMean, arrStd);
        break;
    case InterpolationTactics::GPU_NEAREST_CENTER:
        nearest_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dim_grid, dim_block>>>
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, scale, scale, arrMean, arrStd);
        break;
    case InterpolationTactics::GPU_BILINEAR:
        bilinear_BGR2RGB_nhwc2nchw_norm_kernel 
                <<<dim_grid, dim_block>>> 
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, fScaledW, fScaledH, arrMean, arrStd);
        break;
    case InterpolationTactics::GPU_BILINEAR_CENTER:
        bilinear_BGR2RGB_nhwc2nchw_shift_norm_kernel 
                <<<dim_grid, dim_block>>> 
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, scale, scale, arrMean, arrStd);
        break;
    default:
        LOGE("ERROR: Wrong GPU resize InterpolationTactics selected. Program terminated");
        exit(1);
    }
}

void ResizeBilinearGpu(
    float* pDeviceTar, uint8_t* pDeviceSrc, 
    int iTarW, int iTarH, 
    int iSrcW, int iSrcH, 
    InterpolationTactics eTac) 
{
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(iTarW / 32 + 1, iTarH / 32 + 1, 1);

    float fScaledH = (float)iSrcH / iTarH;
    float fScaledW = (float)iSrcW / iTarW;
    float scale = (fScaledH > fScaledW ? fScaledH : fScaledW);

    switch (eTac) {
    case InterpolationTactics::GPU_NEAREST:
        nearest_BGR2RGB_nhwc2nchw_kernel <<<dim_grid, dim_block>>>
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, fScaledW, fScaledH);
        break;
    case InterpolationTactics::GPU_NEAREST_CENTER:
        nearest_BGR2RGB_nhwc2nchw_kernel <<<dim_grid, dim_block>>>
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, scale, scale);
        break;
    case InterpolationTactics::GPU_BILINEAR:
        bilinear_BGR2RGB_nhwc2nchw_kernel <<<dim_grid, dim_block>>> 
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, fScaledW, fScaledH);
        break;
    case InterpolationTactics::GPU_BILINEAR_CENTER:
        bilinear_BGR2RGB_nhwc2nchw_shift_kernel <<<dim_grid, dim_block>>> 
                (pDeviceTar, pDeviceSrc, iTarW, iTarH, iSrcW, iSrcH, scale, scale);
        break;
    case InterpolationTactics::GPU_WARP_AFFINE:
        warpaffine_init(iSrcH, iSrcW, iTarH, iTarW);
        warpaffine_BGR2RGB_kernel <<<dim_grid, dim_block>>> 
                (pDeviceTar, pDeviceSrc, cTransInfo, cAffineMatrix);
        break;
    default:
        LOGE("ERROR: Wrong GPU resize InterpolationTactics selected. Program terminated");
        exit(1);
    }
}