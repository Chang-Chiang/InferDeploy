#include "include/tasks/TaskUtils.h"

float IouCalc(Bbox cBbox1, Bbox cBbox2) {
    auto f_inter_x0 = std::max(cBbox1.m_fX0, cBbox2.m_fX0);
    auto f_inter_y0 = std::max(cBbox1.m_fY0, cBbox2.m_fY0);
    auto f_inter_x1 = std::min(cBbox1.m_fX1, cBbox2.m_fX1);
    auto f_inter_y1 = std::min(cBbox1.m_fY1, cBbox2.m_fY1);

    float f_inter_w = f_inter_x1 - f_inter_x0;
    float f_inter_h = f_inter_y1 - f_inter_y0;

    float f_inter_area = f_inter_w * f_inter_h;
    float f_union_area = (cBbox1.m_fX1 - cBbox1.m_fX0) * (cBbox1.m_fY1 - cBbox1.m_fY0) +
                         (cBbox2.m_fX1 - cBbox2.m_fX0) * (cBbox2.m_fY1 - cBbox2.m_fY0) - f_inter_area;

    return f_inter_area / f_union_area;
}