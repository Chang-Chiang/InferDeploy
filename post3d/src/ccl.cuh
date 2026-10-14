#pragma once

/*
 * GPU 上的三维连通域标记（26 邻接）+ 尺寸过滤。
 * 用于剔除跨层孤立的假阳性（小连通域）。
 */
#include <cstdint>

/// @brief 连通域标记与尺寸过滤的统计结果
struct CclResult {
    int  numComponentsTotal = 0;  // 找到的连通域数（尺寸 >= 1）
    int  numComponentsKept  = 0;  // 通过尺寸过滤后存活的连通域数
    int  largestComponent   = 0;  // 最大连通域的体素数
    long foregroundBefore   = 0;  // 过滤前的前景体素数
    long foregroundAfter    = 0;  // 过滤后的前景体素数
    int  iterations         = 0;  // 并查集 merge/compress 迭代到收敛的次数
};

/// @brief 原地操作：d_mask 是尺寸为 W*H*D 的 0/1 uint8 体数据
///        体素数少于 minSize 的连通域会被清除为背景；
///        keepLargestOnly 为真时只保留最大的那一个连通域
CclResult filterComponents(uint8_t* d_mask, int W, int H, int D, int minSize, bool keepLargestOnly);
