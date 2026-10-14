#pragma once

/*
 * 采用可分离 box 结构元素的三维二值形态学。
 *   闭运算（先膨胀后腐蚀）：弥合层间空隙，填补内部孔洞
 *   开运算（先腐蚀后膨胀）：去除毛刺 / 细弱的伪突出
 * box 结构元素可分离，因此每次三维操作都等价于三个方向的一维 min/max
 * 扫描（每个体素 O(r) 开销，而非 O(r^3)）。三个轴的半径可以不同，
 * 以便沿 Z（切面堆叠方向）拉长结构元素。
 */
#include <cstdint>

/// @brief 对 0/1 uint8 体数据先做闭运算再做开运算（两者各自可单独关闭）
void morphOpenClose(uint8_t* d_mask, int W, int H, int D, int rx, int ry, int rz, bool doClose, bool doOpen);
