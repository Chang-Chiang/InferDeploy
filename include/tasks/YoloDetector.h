#pragma once

#include "include/core/InferTRT.h"
#include "include/tasks/TaskUtils.h"


class YoloDetector : public InferTRT{

public:
    YoloDetector(std::string strOnnxPath, Level cLevel, Params cParams)
        : InferTRT(strOnnxPath, cLevel, cParams) {};

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