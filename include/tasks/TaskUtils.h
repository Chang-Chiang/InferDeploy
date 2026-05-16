#pragma once

#include <opencv2/opencv.hpp>

enum ModelType {
    YOLOV5,
    YOLOV8
};

struct Bbox {
    float     m_fX0, m_fX1, m_fY0, m_fY1;
    float     m_fConfidence;
    bool      m_bFlgRemove;
    int       m_iLabel;
    cv::Rect  m_cvRect;
    cv::Mat   m_matBoxMask;
    cv::Mat   m_matMc;
    
    Bbox() = default;

    Bbox(float fX0, float fY0, float fX1, float fY1, float fConfidence, int iLabel)
        : m_fX0(fX0)
        , m_fY0(fY0)
        , m_fX1(fX1)
        , m_fY1(fY1)
        , m_fConfidence(fConfidence)
        , m_bFlgRemove(false)
        , m_iLabel(iLabel){};

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
    }
};

float IouCalc(Bbox cBbox1, Bbox cBbox2);