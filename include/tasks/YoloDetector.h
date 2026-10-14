#pragma once

#include <memory>
#include "include/core/InferBase.h"
#include "include/tasks/TaskUtils.h"

class YoloDetector : public InferBase {

public:
    YoloDetector(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                 std::shared_ptr<InferBackend> pBackend)
        : InferBase(strOnnxPath, cLevel, cParams, std::move(pBackend)) {};

public:
    virtual void SetupIoBuffers() override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;

private:
    std::vector<Bbox> m_vecBbox;
    int               m_iInputSize;
    int               m_iImgArea;
    int               m_iOutputSize;
    Dims              m_nvInputDims;
    Dims              m_nvOutputDims;

public:
    /// @brief 获取 NMS 后的最终检测框（PostprocessCpu 之后有效）
    /// @return 检测框列表（坐标为原图尺度，置信度为 0~1）
    std::vector<Bbox> GetFinalBboxes() const;
};

// 外部调用的接口
std::unique_ptr<YoloDetector> MakeDetector(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                                           std::shared_ptr<InferBackend> pBackend);
