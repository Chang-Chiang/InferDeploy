#pragma once

#include "include/core/InferTRT.h"
#include "include/tasks/TaskUtils.h"

class YoloMultitask : public InferTRT {

public:
    YoloMultitask(std::string strOnnxPath, Level cLevel, Params cParams)
        : InferTRT(strOnnxPath, cLevel, cParams) { };

public:
    virtual void Setup(void const* pData, std::size_t iSize) override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;
    virtual bool PostprocessGpu() override;

private:
    std::vector<Bbox> m_vecBbox;
    cv::Mat           m_matMask;

    int m_iInputSize; 
    int m_iImgArea;
    int m_iDetectSize;
    int m_iSegmentSize;

    float* m_pDetectMemory[2];
    float* m_pSegmentMemory[2];

    nvinfer1::Dims m_nvDetectDims;
    nvinfer1::Dims m_nvSegmentDims;
};

std::shared_ptr<YoloMultitask> MakeMultitask(
    std::string strOnnxPath, Level cLevel, Params cParams);
