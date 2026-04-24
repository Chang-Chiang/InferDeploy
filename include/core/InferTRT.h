#pragma once

#include <NvInfer.h>
// #include <NvOnnxParser.h>

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

/// @brief TensorRT推理基类，封装了TensorRT的模型构建和推理流程，具体的前处理/后处理由子类实现
class InferTRT {

public:
    /// @brief 推理类构造函数, 主要负责初始化成员变量，构造engine文件路径等
    /// @param strOnnxPath 
    /// @param cLevel 
    /// @param cParams 
    InferTRT(std::string strOnnxPath, Level cLevel, Params cParams); 

    /// @brief 推理类析构函数, 主要负责释放成员变量，释放engine文件路径等
    virtual ~InferTRT() {};

    /// @brief 加载待推理图像
    /// @param strImagePath 
    void LoadImage(std::string strImagePath);

    /// @brief 初始化模型，包括build推理引擎, 分配内存，创建context, 设置bindings
    void InitModel();
    
    /// @brief 推理部分，包括前处理-推理-后处理
    void Inference();

    std::string GetPrec(Precision ePrec);

public:
    bool BuildEngine();
    bool LoadEngine();
    void SavePlan(nvinfer1::IHostMemory& nvPlan);
    void PrintNetwork(nvinfer1::INetworkDefinition &nvNetwork, bool bOptimized);
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
    std::string m_strImagePath;
    std::string m_strOutputPath;
    std::string m_strOnnxPath;
    std::string m_strEnginePath;

    cv::Mat m_matInputImage;
    Params* m_pParams;

    int    m_iWorkspaceSize;
    float* m_pBindings[2];
    float* m_pInputMemory[2];
    float* m_pOutputMemory[2];

    nvinfer1::Dims m_nvInputDims;
    nvinfer1::Dims m_nvOutputDims;
    cudaStream_t   m_cuStream;

    std::shared_ptr<Logger>                       m_pLogger;
    std::shared_ptr<Timer>                        m_pTimer;
    std::shared_ptr<nvinfer1::IRuntime>           m_pRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine>        m_pEngine;
    std::shared_ptr<nvinfer1::IExecutionContext>  m_pContext;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_pNetwork;

};
