#pragma once

#include "include/core/InferTRT.h"

enum ModelType {
    YOLOV5,
    YOLOV8
};

struct Bbox {
    float m_fX0, m_fX1, m_fY0, m_fY1;
    float m_fConfidence;
    bool  m_bFlgRemove;
    int   m_iLabel;
    
    Bbox() = default;
    Bbox(float fX0, float fY0, float fX1, float fY1, float fConfidence, int iLabel)
        : m_fX0(fX0)
        , m_fY0(fY0)
        , m_fX1(fX1)
        , m_fY1(fY1)
        , m_fConfidence(fConfidence)
        , m_bFlgRemove(false)
        , m_iLabel(iLabel){};
};

class YoloDetector : public InferTRT{

public:
    YoloDetector(std::string strOnncPath, Level cLevel, Params cParams)
        : InferTRT(strOnncPath, cLevel, cParams) {};

public:
    virtual void Setup(void const* pData, std::size_t iSize) override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;
    virtual bool PostprocessGpu() override;

private:
    std::vector<Bbox> m_vecBbox;
    int m_iInputSize; 
    int m_iImgArea;
    int m_iOutputSize;
};

// 外部调用的接口
std::shared_ptr<YoloDetector> MakeDetector(
    std::string strOnnxPath, Level cLevel, Params cParams);