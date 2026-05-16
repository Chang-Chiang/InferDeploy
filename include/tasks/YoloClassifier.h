#pragma once

#include <memory>
#include "include/core/InferTRT.h"

class YoloClassifier : public InferTRT {

public:
    YoloClassifier(std::string strOnnxPath, Level cLevel, Params cParams)
        : InferTRT(strOnnxPath, cLevel, cParams) { };

public:
    virtual void Setup(void const* pData, std::size_t iSize) override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;
    virtual bool PostprocessGpu() override;

private:
    float       m_fConfidence;
    std::string m_strLabel;
    int         m_iInputSize; 
    int         m_iImgArea;
    int         m_iOutputSize;
};

std::unique_ptr<YoloClassifier> MakeClassifier(std::string strOnnxPath, Level cLevel, Params cParams);
