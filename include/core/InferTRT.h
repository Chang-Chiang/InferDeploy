#pragma once

#include <NvInfer.h>
#include <memory>

#include <string>

#include "include/utils/Logger.h"
#include "include/utils/Timer.h"
#include "include/common/ModelParam.h"

/// @brief TensorRT 资源释放模板函数
/// @tparam T 
/// @param ptr 
template<typename T>
void DestroyTrtPtr(T* ptr){
    if (ptr) {
        std::string str_type_name = typeid(T).name();
        LOGD("Destroy %s", str_type_name.c_str());
        ptr->destroy(); 
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

/// @brief TensorRT推理基类，封装了TensorRT的模型构建和推理流程，具体的前处理/后处理由子类实现
class InferTRT {

public:
    /// @brief 推理类构造函数, 初始化成员变量
    /// @param strOnnxPath onnx 模型路径
    /// @param cLevel 日志级别
    /// @param cParams 模型构建参数
    InferTRT(std::string strOnnxPath, Level cLevel, Params cParams); 

    /// @brief 推理类析构函数
    virtual ~InferTRT() {};

    /// @brief 加载待推理图像
    /// @param strImagePath 图像路径
    void LoadImage(std::string strImagePath);

    /// @brief 初始化模型，包括build推理引擎, 分配内存，创建context, 设置bindings
    void InitModel();
    
    /// @brief 推理部分，包括前处理-推理-后处理
    void Inference();

    std::string GetPrec(Precision ePrec);

public:
    /// @brief 构建推理引擎，包括解析onnx模型、设置构建参数、构建引擎、序列化引擎等步骤
    /// @return 
    bool BuildEngine();

    /// @brief 加载已有的推理引擎，包括反序列化引擎、创建context、设置bindings等步骤
    /// @return 
    bool LoadEngine();

    /// @brief 保存推理引擎到文件
    /// @param nvPlan 
    void SavePlan(nvinfer1::IHostMemory& nvPlan);

    /// @brief 打印网络结构信息，包括输入输出信息、层信息等
    /// @param nvNetwork 
    /// @param bOptimized 
    void PrintNetwork(nvinfer1::INetworkDefinition &nvNetwork, bool bOptimized);

    /// @brief 执行推理，调用 context 的 enqueueV2 函数执行推理，并记录推理时间
    /// @return 
    bool EnqueueBindings();

/// 不同任务的实现不一样，定义为纯虚函数，子类具体实现
public:
    /// @brief 分配 host/device 的 memory(不同任务输入输出不一样)、bindings、以及创建推理所需要的上下文
    /// @param pData 
    /// @param iSize 
    virtual void Setup(void const* pData, std::size_t iSize) = 0;

    virtual void ResetTask() = 0;

    virtual bool PreprocessCpu()      = 0;
    virtual bool PreprocessGpu()      = 0;
    virtual bool PostprocessCpu()     = 0;
    virtual bool PostprocessGpu()     = 0;

public:
    std::string m_strImagePath;    // 输入图像路径
    std::string m_strOutputPath;   // 输出路径
    std::string m_strOnnxPath;     // onnx 模型路径
    std::string m_strEnginePath;   // 引擎保存路径

    cv::Mat m_matInputImage;  // 输入图像的 Mat 对象
    Params m_cParams;

    int    m_iWorkspaceSize;    // TensorRT engine构建时的工作空间大小，单位为字节
    float* m_pBindings[2];      // bindings 指针，用于存储输入和输出的内存地址
    float* m_pInputMemory[2];   // 输入的 host/device 内存指针，m_pInputMemory[0] 是 host 内存，m_pInputMemory[1] 是 device 内存
    float* m_pOutputMemory[2];  // 输出的 host/device 内存指针，m_pOutputMemory[0] 是 host 内存，m_pOutputMemory[1] 是 device 内存

    nvinfer1::Dims m_nvInputDims;   // 输入维度
    nvinfer1::Dims m_nvOutputDims;  // 输出维度
    cudaStream_t   m_cuStream;      // cuda 流

    Logger                                        m_cLogger;
    Timer                                         m_cTimer;
    TrtUniquePtr<nvinfer1::IRuntime>              m_pRuntime;
    TrtUniquePtr<nvinfer1::ICudaEngine>           m_pEngine;
    TrtUniquePtr<nvinfer1::IExecutionContext>     m_pContext;
    TrtUniquePtr<nvinfer1::INetworkDefinition>    m_pNetwork;

};
