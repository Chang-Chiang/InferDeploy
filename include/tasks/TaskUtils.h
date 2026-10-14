#pragma once

#include <opencv2/opencv.hpp>

/// @brief 模型类型
enum ModelType {
    YOLOV5,
    YOLOV8
};

/// @brief 检测框结构体
struct Bbox {
    float    m_fX0, m_fX1, m_fY0, m_fY1;  // 检测框坐标，左上角 (X0, Y0)，右下角 (X1, Y1)
    float    m_fConfidence;               // 检测框置信度
    bool     m_bFlgRemove;                // 检测框是否被删除标识
    int      m_iLabel;                    // 检测框类别
    cv::Rect m_cvRect;                    // 检测框对应的矩形框
    cv::Mat  m_matBoxMask;                // 检测框对应的掩码，用于分割任务
    cv::Mat  m_matMc;                     // 检测框对应的 MC，用于分割任务

    Bbox() = default;

    Bbox(float fX0, float fY0, float fX1, float fY1, float fConfidence, int iLabel)
        : m_fX0(fX0)
        , m_fY0(fY0)
        , m_fX1(fX1)
        , m_fY1(fY1)
        , m_fConfidence(fConfidence)
        , m_bFlgRemove(false)
        , m_iLabel(iLabel) {};

    Bbox(float fX0, float fY0, float fX1, float fY1, float fConfidence, int iLabel, cv::Mat matImg)
        : m_fX0(fX0)
        , m_fY0(fY0)
        , m_fX1(fX1)
        , m_fY1(fY1)
        , m_fConfidence(fConfidence)
        , m_bFlgRemove(false)
        , m_iLabel(iLabel) {

        m_cvRect = cv::Rect_<int>(m_fX0, m_fY0, m_fX1 - m_fX0, m_fY1 - m_fY0);

        // 如果rect越界，调整rect的大小
        if (m_cvRect.x + m_cvRect.width > matImg.cols) {
            m_cvRect.width = matImg.cols - m_cvRect.x;
        }
        if (m_cvRect.y + m_cvRect.height > matImg.rows) {
            m_cvRect.height = matImg.rows - m_cvRect.y;
        }

        m_cvRect.x = m_cvRect.x < 0 ? 0 : m_cvRect.x;
        m_cvRect.y = m_cvRect.y < 0 ? 0 : m_cvRect.y;

        // 防御性保护：量化模型可能输出坐标倒置/异常的框，保证 ROI 有效
        if (m_cvRect.width < 1) {
            m_cvRect.width = 1;
        }
        if (m_cvRect.height < 1) {
            m_cvRect.height = 1;
        }
    }
};

float IouCalc(Bbox cBbox1, Bbox cBbox2);