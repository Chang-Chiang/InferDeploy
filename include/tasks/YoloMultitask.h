#pragma once

#include <memory>
#include "include/core/InferBase.h"
#include "include/tasks/TaskUtils.h"

class YoloMultitask : public InferBase {

public:
    YoloMultitask(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                  std::shared_ptr<InferBackend> pBackend)
        : InferBase(strOnnxPath, cLevel, cParams, std::move(pBackend)) {
        m_pDetectMemory[0]  = nullptr;
        m_pDetectMemory[1]  = nullptr;
        m_pSegmentMemory[0] = nullptr;
        m_pSegmentMemory[1] = nullptr;
    };

    ~YoloMultitask() override {
        ReleaseExtraBuffers();
    }

public:
    virtual void SetupIoBuffers() override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;

private:
    std::vector<Bbox> m_vecBbox;
    cv::Mat           m_matMask;

    int m_iInputSize;
    int m_iImgArea;
    int m_iDetectSize;
    int m_iSegmentSize;

    float* m_pDetectMemory[2];
    float* m_pSegmentMemory[2];

    Dims m_nvDetectDims;
    Dims m_nvSegmentDims;

    std::string m_strDetectName;   // 检测头输出 tensor 名
    std::string m_strSegmentName;  // 分割 proto 头输出 tensor 名

public:
    /// @brief 获取 NMS 后的最终检测框（PostprocessCpu 之后有效）
    /// @return 检测框列表（坐标为原图尺度，置信度为 0~1）
    std::vector<Bbox> GetFinalBboxes() const;

protected:
    /// @brief 双输出：按 tensor 名返回检测/分割各自的缓冲
    float* GetOutputHostPtr(int i) override;
    float* GetOutputDevicePtr(int i) override;

    /// @brief 释放自有的检测/分割双输出缓冲
    void ReleaseExtraBuffers() override;
};

std::unique_ptr<YoloMultitask> MakeMultitask(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                                             std::shared_ptr<InferBackend> pBackend);
