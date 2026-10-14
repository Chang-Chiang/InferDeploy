#pragma once

/*
 * 对二值掩码做可分离的三维高斯平滑，随后重新阈值化。
 * 先把 0/1 掩码转成 float，再沿 X、Y、Z 三个方向分别与一维高斯核卷积
 * （可分离 => 每个体素 O(k) 开销），最后在 threshold（默认 0.5）处重新
 * 二值化，得到抑制了边缘噪声的规整二值掩码。
 */
#include <cstdint>

/// @brief 对 d_mask 原地操作。sigma 以体素为单位；threshold 取值 [0,1]
void gaussianSmoothMask(uint8_t* d_mask, int W, int H, int D, float sigma, float threshold);
