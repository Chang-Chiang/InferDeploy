#pragma once

#include <memory>
#include "include/core/InferBase.h"

/// @brief UNet 医学图像分割任务（肝脏 CT/MR 切片）
/// 模型: liver-unet.onnx，输入 1x1x256x256（灰度 /255），输出 1x1x256x256
/// 预处理: 灰度读取 → 256x256 缩放 → /255（CPU 路径 cv::resize，GPU 路径 ResizeGrayGpu）
/// 后处理: 输出阈值二值化 → 上采样回原图 → 叠加红色 mask 并保存
///
/// @note 当前部署的 liver-unet.onnx 图末层不含 Sigmoid，输出是 **logits**（实测-19.7~+10.2），
///       因此下面的 0.5 阈值实际等价于 sigmoid > 0.6225（比"概率 0.5"更严）。
///       LiTS 留出切片实测：该阈值比 logits > 0 的 Dice 高约 1.5 pp（模型偏过分割）。
///       **更换模型时必须先确认输出是 logits 还是概率，再定阈值**，否则阈值语义会静默漂移。
class UnetSegmenter : public InferBase {

public:
    UnetSegmenter(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
                  std::shared_ptr<InferBackend> pBackend)
        : InferBase(strOnnxPath, cLevel, cParams, std::move(pBackend)) {};

public:
    virtual void SetupIoBuffers() override;
    virtual void ResetTask() override;
    virtual bool PreprocessCpu() override;
    virtual bool PreprocessGpu() override;
    virtual bool PostprocessCpu() override;

private:
    cv::Mat m_matMask;        // 0/255 二值 mask（原图尺寸）
    int m_iInputSize;         // 输入缓冲字节数
    int m_iOutputSize;        // 输出缓冲字节数
    Dims m_nvInputDims;
    Dims m_nvOutputDims;
    bool m_bSaveDebug = true; // 是否在 PostprocessCpu 中落盘结果图（体数据批量分割时可关闭）

public:
    /// @brief 获取分割结果（PostprocessCpu 之后有效）
    /// @return 0/255 二值 mask（原图尺寸）
    cv::Mat GetMask() const { return m_matMask; }

    /// @brief 体数据批量分割时关闭逐层落盘，由调用方统一写 3D 输出
    void SetSaveDebug(bool bEnable) { m_bSaveDebug = bEnable; }
};

std::unique_ptr<UnetSegmenter> MakeUnetSegmenter(
    const std::string& strOnnxPath, Level cLevel, const Params& cParams,
    std::shared_ptr<InferBackend> pBackend);
