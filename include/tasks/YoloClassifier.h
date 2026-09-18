#pragma once

#include <memory>
#include <utility>
#include "include/core/InferBase.h"

class YoloClassifier : public InferBase {

public:
    YoloClassifier(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                   std::shared_ptr<InferBackend> pBackend)
        : InferBase(strOnnxPath, cLevel, cParams, std::move(pBackend)) { };

public:
    virtual void SetupIoBuffers() override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;

private:
    float       m_fConfidence;
    int         m_iTop1Index;
    std::string m_strLabel;
    int         m_iInputSize;
    int         m_iImgArea;
    int         m_iOutputSize;
    Dims        m_nvInputDims;
    Dims        m_nvOutputDims;

public:
    /// @brief 获取分类结果（PostprocessCpu 之后有效）
    /// @return first: top-1 类别索引，second: 置信度（0~1）
    std::pair<int, float> GetResult() const {
        return {m_iTop1Index, m_fConfidence};
    }
};

std::unique_ptr<YoloClassifier> MakeClassifier(const std::string& strModelPath, Level cLevel, const Params& cParams,
                                               std::shared_ptr<InferBackend> pBackend);
