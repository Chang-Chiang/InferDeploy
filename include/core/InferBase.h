#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "include/utils/Logger.h"
#include "include/utils/Timer.h"
#include "include/common/ModelParam.h"

/// @brief 输入/输出张量元信息（由后端在 LoadModel 时填充）
struct IoMeta {
    std::string strName;  // 张量名
    Dims        dims;     // 张量维度（NCHW）
};

class InferBase;  // 前向声明

/// @brief 推理后端接口，由 TrtBackend / OvBackend 实现
/// 后端负责：模型加载（构建/反序列化/编译）、缓冲绑定、单次推理执行
class InferBackend {
public:
    virtual ~InferBackend() = default;

    /// @brief 加载/构建模型，填充输入输出元信息（含按输入 shape 回写 Params.m_cImgInfo）
    virtual bool LoadModel(InferBase* pOwner) = 0;

    /// @brief 将任务类 SetupIoBuffers 分配好的缓冲绑定到模型
    /// （缓冲地址通过 InferBase 的 GetInput*/GetOutput*Ptr 虚函数获取）
    virtual bool BindBuffers(InferBase* pOwner) = 0;

    /// @brief 执行一次推理（预处理已完成，输入数据已就绪）
    virtual bool Enqueue(InferBase* pOwner) = 0;

    /// @brief 释放后端资源
    virtual void Release() = 0;
};

/// @brief 推理抽象基类：封装通用的「前处理 → 推理 → 后处理」执行流，
///        与具体推理后端（TensorRT/OpenVINO）解耦；
///        具体任务（分类/检测/多任务）继承本类，只需实现前后处理与缓冲分配
class InferBase {
public:
    /// @brief 构造函数
    /// @param strOnnxPath onnx 模型路径
    /// @param cLevel 日志级别
    /// @param cParams 模型构建参数
    /// @param pBackend 推理后端（TrtBackend/OvBackend）
    InferBase(const std::string& strOnnxPath, Level cLevel, const Params& cParams,
              std::shared_ptr<InferBackend> pBackend);

    /// @brief 析构函数：释放缓冲与 CUDA 资源
    virtual ~InferBase();

    /// @brief 加载待推理图像
    void LoadImage(const std::string& strImagePath);

    /// @brief 初始化模型：LoadModel → SetupIoBuffers → BindBuffers；已有上下文时仅重置任务状态
    void InitModel();

    /// @brief 推理：前处理 → 后端推理 → 后处理
    void Inference();

    std::string GetPrec(Precision cPrec);

    /// @brief 任务类实现的接口：按 IoMeta 分配输入/输出缓冲（后端无关）
    virtual void SetupIoBuffers() = 0;

    virtual void ResetTask() = 0;

    virtual bool PreprocessCpu()  = 0;
    virtual bool PreprocessGpu()  = 0;
    virtual bool PostprocessCpu() = 0;

    /// @brief GPU 后处理，默认委托给 PostprocessCpu()，子类可覆盖实现真正的 GPU 后处理
    virtual bool PostprocessGpu() {
        return PostprocessCpu();
    }

protected:
    /// @brief 释放通用 host/device 缓冲与 CUDA stream
    void CleanupBuffers();

    /// @brief 任务类自有缓冲（如 Multitask 的双输出）的释放钩子，析构时调用
    virtual void ReleaseExtraBuffers() {
    }

public:
    // ---- 缓冲地址访问器（后端 BindBuffers/Enqueue 时使用；多输出任务类可重写） ----
    virtual float* GetInputHostPtr(int i) {
        return m_pInputMemory[0];
    }

    virtual float* GetInputDevicePtr(int i) {
        return m_pInputMemory[1];
    }

    virtual float* GetOutputHostPtr(int i) {
        return m_pOutputMemory[0];
    }

    virtual float* GetOutputDevicePtr(int i) {
        return m_pOutputMemory[1];
    }

public:
    std::string m_strImagePath;   // 输入图像路径
    std::string m_strOutputPath;  // 输出路径
    std::string m_strOnnxPath;    // onnx 模型路径

    cv::Mat m_matInputImage;  // 输入图像的 Mat 对象
    Params  m_cParams;

    // 通用输入/输出缓冲：[0] host（pinned），[1] device
    float* m_pInputMemory[2];
    float* m_pOutputMemory[2];

    // I/O 元信息（由后端 LoadModel 填充，供任务类 SetupIoBuffers 使用）
    std::vector<IoMeta> m_vecInputs;
    std::vector<IoMeta> m_vecOutputs;

    cudaStream_t m_cuStream;  // cuda 流（预处理异步拷贝/后端推理共用）

    Logger m_cLogger;
    Timer  m_cTimer;

    std::shared_ptr<InferBackend> m_pBackend;  // 推理后端
};
