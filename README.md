# InferDeploy
深度学习模型推理部署

## 背景

针对深度学习任务，基于 PyTorch 进行训练后获得对应模型文件后，在端侧基于 TensorRT/OpenVINO 进行部署。

而在模型的推理部署中，针对不同任务，模型的加载编译流程基本一致，主要差异在于输入模型前后数据的预处理、后处理流程。

该项目框架主要对模型加载编译流程进行封装，针对具体的任务，只需在此基础上重写预处理、后处理流程即可完成部署。

## 使用

```bash
# 项目工程下载至本地
git clone https://github.com/Chang-Chiang/InferDeploy

# 项目编译
./build.sh

# 运行 sample
cd ./build/bin
./yolo_detector
```

## 整体架构总览

### 工程化

- **代码可读性**：通过 **`Worker` 类**对外仅暴露极简的核心能力，
  - 主函数中仅需 `创建 Worker 实例 → 加载图片 → 执行推理` 三步即可完成端到端推理，
  - 完全屏蔽底层 TensorRT 引擎构建、内存管理、CUDA 调度的复杂细节
  - 内部按职责拆分模块，`Worker` 下分设 `Classifier` / `Detector` / `Segmenter` 等独立任务单元，职责边界清晰，代码逻辑分层明确
- **代码可复用性**：通过 **`InferTRT` 类**提炼 TensorRT 部署的全流程通用范式，
  - 模型构建全流程 `builder → network → config → parser → serialize`
  - 推理执行全流程 `load → deserialize → engine → context → enqueue`
  - 内存管理全流程 `host/device内存分配与释放`
  - 以及所有任务通用的 `前处理 → DNN推理 → 后处理` 执行流
- **高可扩展性**：通过 `Classifier` / `Detector` / `Segmenter` 类，继承 `InferTRT` 类后，重写预处理/后处理流程实现
  - 采用核心框架与具体任务完全解耦的模块化设计，新增任务时仅需继承基类实现对应接口，无需修改框架核心代码，实现最小化改动扩展
  - 不仅预留了分类、检测、分割、关键点检测等单任务的扩展接口，还支持多阶段模型（先检测再分类）、视频流异步处理、多线程推理、多任务联合推理等进阶场景的无缝扩展
- **内存与调用安全性**：通过 `shared_ptr` 智能指针实现，
  - 基于 `shared_ptr` 智能指针自动管理内存生命周期，避免手动释放内存导致的遗漏与重复释放
  - 采用 **RAII（资源获取即初始化）** 机制，将引擎创建、内存分配、CUDA 上下文初始化等资源申请封装在类构造函数中，资源释放封装在析构函数中，确保资源生命周期与对象完全绑定，从机制上规避资源泄漏风险
- **可调试性**：
  - 实现了分级日志系统`Logger`，支持`VERBOSE`/`INFO`/`WARNING`/`ERROR`等多级别日志控制，可通过入参灵活调整日志输出粒度
  - 在模型构建、引擎加载、前后处理、推理执行等关键节点埋点调试信息，可快速定位 NMS 输出、网络层数据布局（NCHW/NHWC）、TensorRT 优化后网络结构等问题，大幅提升开发调试效率

```mermaid
flowchart TD
    A[Main函数/业务调用] --> B[统一接口层 Worker 类<br> 提供用户调用接口]
    
    B --> C1[分类任务<br>Classifier]
    B --> C2[检测任务<br>Detector]
    B --> C3[分割任务<br>Segmenter]
    B --> C4[扩展任务<br>Pose Estimation/...]
    
    C1 & C2 & C3 & C4 --> D[核心框架层 InferTRT 类]

    D --> D1[内存管理模块<br>智能指针/RAII]
    D --> D2[分级日志系统<br>Logger]
    D --> E[底层依赖层]
    
    E --> E1[TensorRT<br>引擎构建/推理执行]
    E --> E2[CUDA<br>内存管理/核函数调度]
    
    E --> E3[OpenCV<br>图像读取/处理/结果绘制]
```



### 继承关系

- **Classifier、Detector、Multitask** 都继承自 **InferTRT**，体现了 "多态" 设计原则

- **Logger** 继承自 TensorRT 的 **ILogger** 接口，实现了自定义的日志功能

```mermaid
classDiagram
Classifier "1" --|> "1" Model : 继承
Detector "1" --|> "1" Model : 继承
Multitask "1" --|> "1" Model : 继承
Logger "1" --|> "1" ILogger: 继承
```



### 组合关系

- **InferTRT** 持有 Params、Logger、Timer 等核心组件，以及 TensorRT 的所有运行时组件

- **Worker** 持有 Logger、Params 以及三种任务处理器（Classifier/Detector/Multitask）

- **Detector 和 Multitask** 都持有多个 bbox 对象

```mermaid
classDiagram
Worker "1" --* "1" Logger : 组合
Worker "1" --* "1" Params : 组合
Worker "1" --* "1" Classifier : 组合
Worker "1" --* "1" Detector : 组合
Worker "1" --* "1" Multitask : 组合
```

```mermaid
classDiagram
Model "1" --* "1" Params : 组合
Model "1" --* "1" Logger : 组合
Model "1" --* "1" Timer : 组合
Model "1" --* "1" IRuntime : 组合
Model "1" --* "1" ICudaEngine : 组合
Model "1" --* "1" IExecutionContext : 组合
Model "1" --* "1" INetworkDefinition : 组合

Detector "1" --* "*" bbox : 组合
Multitask "1" --* "*" bbox : 组合
```



## 模型框架介绍

### 用户调用接口

```c++
// 指定参数：模型文件路径、日志等级、模型构建参数
// 创建 worker
auto p_worker = CreateWorker(str_onnx_path, e_level, c_params);

// 指定待推理图片, 进行推理
p_worker->Inference(str_img_path);
```

### Worker 类，提供外部调用接口

- 作用：保证代码的可读性，只对用户可见必要接口
- 提供接口：
  - 构造函数：根据参数中指定任务类型，构建对应模型
  - 执行推理：读取图像，进行推理

```mermaid
classDiagram
class Worker {
    - m_pLogger : std::shared_ptr<Logger>
    - m_cParams : Params
    - m_pClassifier : std::shared_ptr<Classifier>
    - m_pDetector : std::shared_ptr<Detector>
    - m_pMultitask : std::shared_ptr<Multitask>
    - m_vecScore : std::vector<float>
    - m_vecBbox : std::vector<Bbox>

    + Worker(strOnnxPath : std::string, cLevel : Level, cParams : Params)
    + Inference(strImagePath : std::string)
}
```

### InferTRT 类，核心框架

- 作用：
  - TensorRT 推理基类，封装 TensorRT 的模型构建和推理流程
  - 提供预处理、后处理接口，具体任务类继承后重写
- 提供接口：
  - 构造函数
  - 图像加载
  - 模型初始化 
    - 加载模型：`.engine` 文件存在则直接加载
    - 构建模型：`.engine` 文件不存在，则从 `.onnx` 文件开始构建
  - 执行推理：
    - 预处理（纯虚函数）
    - 推理
    - 后处理（纯虚函数）
  - 保存模型文件，`.engine` 文件
  - 打印网络结构
  - 获取网络量化精度

```mermaid
classDiagram
class Model {
    - m_strImagePath : std::string
    - m_strOutputPath : std::string
    - m_strOnnxPath : std::string
    - m_strEnginePath : std::string
    - m_matInputImage : cv::Mat
    - *m_cParams : Params
    - m_iWorkspaceSize : int
    - *m_pBindings[3] : float
    - *m_pInputMemory[2] : float
    - *m_pOutputMemory[2] : float
    - m_nvInputDims : nvinfer1::Dims
    - m_nvOutputDims : nvinfer1::Dims
    - m_cuStream : cudaStream_t
    - m_pLogger : std::shared_ptr<Logger>
    - m_pTimer : std::shared_ptr<Timer>
    - m_pRuntime : std::shared_ptr<nvinfer1::IRuntime>
    - m_pEngine : std::shared_ptr<nvinfer1::ICudaEngine>
    - m_pContext : std::shared_ptr<nvinfer1::IExecutionContext>
    - m_pNetwork : std::shared_ptr<nvinfer1::INetworkDefinition>

    + InferTRT(strOnnxPath : std::string, cLevel : Level, cParams : Params)
    + ~InferTRT()
    + LoadImage(strImagePath : std::string)
    + InitModel() 
    + Inference()
    + BuildEngine() bool
    + LoadEngine() bool
    + SavePlan(nvinfer1::IHostMemory& nvPlan)
    + PrintNetwork(nvinfer1::INetworkDefinition &nvNetwork, bool bOptimized)
    + EnqueueBindings() bool
    + GetPrec(Precision ePrec) std::string
    
    + virtual void Setup(void const* pData, std::size_t iSize) = 0
    + virtual void ResetTask() = 0
    + virtual bool PreprocessCpu() = 0
    + virtual bool PreprocessGpu() = 0
    + virtual bool PostprocessCpu() = 0
    + virtual bool PostprocessGpu() = 0
}
```

### Classifier 类，继承自 InferTRT，具体分类任务类

- 作用：
  - 继承自 `InferTRT` 推理基类，复用模型构建及推理流程
  - 执行分类任务的预处理/后处理
- 提供接口：
  - 构造函数
  - 设置
  - 任务重置
  - 预处理
  - 后处理

```mermaid
classDiagram
class Classifier {
    - m_fConfidence : float
    - m_strLabel : std::string
    - m_iInputSize : int
    - m_iImgArea : int
    - m_iOutputSize : int

    + Classifier(strOnnxPath : std::string, cLevel : Level, cParams : Params)
    + Setup(*pData : void const, iSize : std::size_t)
    + ResetTask()
    + PreprocessCpu()
    + PreprocessGpu()
    + PostprocessCpu()
    + PostprocessGpu()
}
```

### Detector 类，继承自 InferTRT，具体检测任务类

- 作用：
  - 继承自 `InferTRT` 推理基类，复用模型构建及推理流程
  - 执行检测任务的预处理/后处理
- 提供接口：
  - 构造函数
  - 设置
  - 任务重置
  - 预处理
  - 后处理

```mermaid
classDiagram
class Detector {
    - m_vecBbox : std::vector<Bbox>
    - m_iInputSize : int
    - m_iImgArea : int
    - m_iOutputSize : int 

    + Detector(strOnnxPath : std::string, cLevel : Level, cParams : Params)
    + Setup(*pData : void const, iSize : std::size_t)
    + ResetTask()
    + PreprocessCpu()
    + PreprocessGpu()
    + PostprocessCpu()
    + PostprocessGpu()
}
```

### MultiTask 类，继承自 InferTRT，多任务类

- 作用：
  - 继承自 `InferTRT` 推理基类，复用模型构建及推理流程
  - 执行检测/分割多任务的预处理/后处理
- 提供接口：
  - 构造函数
  - 设置
  - 任务重置
  - 预处理
  - 后处理

```mermaid
classDiagram
class Multitask {
    - m_vecBbox : std::vector<Bbox>
    - m_matMask : cv::Mat
    - m_iInputSize : int
    - m_iImgArea : int
    - m_iDetectSize : int 
    - m_iSegmentSize : int
    - *m_pDetectMemory[2] : float 
    - *m_pSegmentMemory[2] : float 
    - m_nvDetectDims : nvinfer1::Dims
    - m_nvSegmentDims : nvinfer1::Dims

    + Multitask(strOnnxPath : std::string, cLevel : Level, cParams : Params)
    + Setup(*pData : void const, iSize : std::size_t)
    + ResetTask()
    + PreprocessCpu()
    + PreprocessGpu()
    + PostprocessCpu()
    + PostprocessGpu()
}
```

### Logger 类，继承自 `nvinfer1::ILogger`，日志记录

- 作用：
  - 继承 `nvinfer1::ILogger`，重写 `log` 函数
  - 提供分级日志功能，支持 FATAL、ERROR、WARN、INFO、VERB、DEBUG 六个级别，并使用颜色编码来区分不同级别的日志输出
- 提供接口：
  - 构造函数
  - 日志函数，重写 `nvinfer1::ILogger` 的  `log` 函数

```mermaid
classDiagram
class Logger {
    - m_eLevel : static Level
    - m_nvSeverity : Severity

    + Logger()
    + Logger(eLevel : Level)
    + log(nvSeverity : Severity, *msg : const char)
    + static void __log_info(eLevel : Level, *format : const char, ...)
    + GetSeverity(eLevel : Level) GetLevel
    + GetLevel(nvSeverity : Severity) Level
}
```

### Timer 类，耗时测量

- 作用：
  - 使用 C++11 的 chrono 库进行 CPU 计时，使用 CUDA 事件进行 GPU 计时
  - 提供 CPU 和 GPU 的精确计时功能，支持纳秒、微秒、毫秒、秒等多种时间单位
- 提供接口：

```mermaid
classDiagram
class Timer {
    - m_pointCpuStart : std::chrono::time_point<std::chrono::high_resolution_clock>
    - m_pointCpuStop : std::chrono::time_point<std::chrono::high_resolution_clock>
    - m_eventGpuStart : cudaEvent_t
    - m_eventGpuStop : cudaEvent_t
    - m_fTimeElasped : float
    - m_strTimeMsg : std::vector<std::string>

    + Timer()
    + ~Timer()
    + StartCpu()
    + StopCpu()
    + StopCpu(strMsg : std::string)
    + StartGpu()
    + StopGpu()
    + StopGpu(strMsg : std::string)
    + Init()
    + Show()
}
```

## 数据结构与参数设置

### 任务类别

```c++
enum TaskType {
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
    MULTITASK
};
```

### 设备类别

```c++
enum Device {
    CPU,
    GPU
};
```

### 量化精度

```c++
enum Precision {
    FP32,
    FP16,
    INT8
};
```

### 预处理策略

```c++
enum class InterpolationTactics : int32_t {
    CPU_NEAREST         = 0,
    CPU_BILINEAR        = 1,
    GPU_NEAREST         = 2,
    GPU_NEAREST_CENTER  = 3,
    GPU_BILINEAR        = 4,
    GPU_BILINEAR_CENTER = 5,
    GPU_WARP_AFFINE     = 6,
};
```

### 日志级别

```c++
enum class Level : int32_t {
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    VERB  = 4,
    DEBUG = 5
};
```

### 模型构建参数

```c++
struct Params {
    Device               m_eDevice              = GPU;
    int                  m_iNumCls              = 1000;
    InterpolationTactics m_eInterpolationTactic = InterpolationTactics::GPU_BILINEAR;
    ImageInfo            m_cImgInfo             = {224, 224, 3};
    TaskType             m_eTask                = CLASSIFICATION;
    int                  m_iWorkspaceSize       = WORKSPACESIZE;
    Precision            m_ePrec                = FP32;
};
```

## 参考

- 韩君，《CUDA与TensorRT部署实战课程》
