#pragma once

#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

#include "include/core/InferBase.h"

/// @brief TensorRT 资源释放模板函数（TensorRT 10+ 已移除 destroy()，直接 delete）
template<typename T>
void DestroyTrtPtr(T* ptr){
    if (ptr) {
        std::string str_type_name = typeid(T).name();
        LOGD("Destroy %s", str_type_name.c_str());
        delete ptr;
    };
}

template<typename T>
struct TrtDeleter {
    void operator()(T* ptr) const {
        DestroyTrtPtr(ptr);
    }
};

template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

/// @brief TensorRT 推理后端：封装模型构建（onnx→engine）、加载、缓冲绑定与推理执行
class TrtBackend : public InferBackend {
public:
    bool LoadModel(InferBase* pOwner) override;
    bool BindBuffers(InferBase* pOwner) override;
    bool Enqueue(InferBase* pOwner) override;
    void Release() override;

private:
    /// @brief 构建推理引擎：解析 onnx → 设置精度/量化/校准 → 序列化保存
    bool BuildEngine(InferBase* pOwner, nvinfer1::INetworkDefinition& nvNetwork);

    /// @brief 从已有 engine 文件反序列化加载
    bool LoadEngine(InferBase* pOwner);

    /// @brief 由序列化 plan 创建 Runtime/Engine/Context 并收集 I/O 元信息
    void SetupRuntime(InferBase* pOwner, void const* pData, std::size_t iSize);

    /// @brief 保存 engine 到文件
    void SavePlan(InferBase* pOwner, nvinfer1::IHostMemory& nvPlan);

    /// @brief 打印网络结构（优化前/后）
    void PrintNetwork(InferBase* pOwner, nvinfer1::INetworkDefinition& nvNetwork, bool bOptimized);

private:
    TrtUniquePtr<nvinfer1::IRuntime>           m_pRuntime;
    TrtUniquePtr<nvinfer1::ICudaEngine>        m_pEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext>  m_pContext;
    std::string m_strEnginePath;   // 引擎保存路径（按 onnx 路径 + 精度后缀生成）
};
