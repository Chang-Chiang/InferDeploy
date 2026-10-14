#pragma once

#ifdef ENABLE_OPENVINO

    #include <memory>
    #include <string>
    #include <vector>

    #include <openvino/openvino.hpp>

    #include "include/core/InferBase.h"

/// @brief OpenVINO 推理后端：读取 ONNX → 编译模型 → 推理
/// 说明：
///   - 模型设备固定为 "CPU"（在无 Intel GPU 的 NVIDIA 机器上 OpenVINO GPU 插件不可用；
///     Intel GPU 环境可改为 "GPU"）；
///   - 预处理仍复用 CUDA 核函数/CPU 流程：Device=GPU 时推理前将输入 D2H，
///     推理后将输出 H2D，保证与任务类通用的后处理拷贝逻辑兼容。
class OvBackend : public InferBackend {
public:
    bool LoadModel(InferBase* pOwner) override;
    bool BindBuffers(InferBase* pOwner) override;
    bool Enqueue(InferBase* pOwner) override;
    void Release() override;

private:
    std::shared_ptr<ov::Model> m_pModel;
    ov::CompiledModel          m_compiledModel;
    ov::InferRequest           m_inferRequest;

    /// @brief IoMeta → ov::Shape
    static ov::Shape ToOvShape(const Dims& cDims);

    /// @brief 计算张量元素个数
    static std::size_t ElementsCount(const Dims& cDims);
};

#endif  // ENABLE_OPENVINO
