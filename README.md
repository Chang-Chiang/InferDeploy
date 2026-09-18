# InferDeploy
深度学习模型推理部署

## 背景

针对深度学习任务，基于 PyTorch 进行训练后获得对应模型文件后，在端侧基于 TensorRT/OpenVINO 进行部署。

而在模型的推理部署中，针对不同任务，模型的加载编译流程基本一致，主要差异在于输入模型前后数据的预处理、后处理流程。

该项目框架主要对模型加载编译流程进行封装，针对具体的任务，只需在此基础上重写预处理、后处理流程即可完成部署。

## 特性总览

- **多后端推理**：同一套任务代码（前/后处理）可运行于 TensorRT（GPU）与 OpenVINO（CPU）双后端，`Params.m_eBackend` 一键切换
- **多任务支持**：图像分类、目标检测、检测+分割多任务、医学图像肝脏分割（2D UNet）
- **体数据分割**：完整 CT volume 逐切面批量推理 + CUDA 3D 后处理流水线（3D CCL → 形态学开闭 → 高斯平滑），肝脏 3D 分割的端到端实测记录（含训练/导出、Dice 评估、跨后端一致性、跨域失效分析）见 [`docs/3D肝脏分割部署流程.md`](./docs/3D肝脏分割部署流程.md)
- **量化评估**：modelopt 显式量化（Q/DQ）+ FP32/INT8 精度对比评估工具（匹配率/IoU/置信度差 + 验收线判定），目标检测的端到端实测记录见 [`docs/目标检测部署流程.md`](./docs/目标检测部署流程.md)
- **工程化**：RAII 资源管理、智能指针、分级日志、CPU/GPU 双计时、模型构建全流程封装

## 环境要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| CUDA | 12.4+ | 含 cudart/cublas |
| TensorRT | 10.16 | 支持 Blackwell（sm_120）；`deps/TensorRT-10.16/usr` |
| OpenVINO | 2026.x | 可选，`ENABLE_OPENVINO=1` 启用；`deps/OpenVINO/usr` |
| OpenCV | 4.x | 系统安装 |
| 编译器 | 支持 C++17 的 g++、nvcc | |

> GPU 架构：`config/Makefile.config` 中 `ARCH` 需与本机 GPU 匹配；pip 版 TensorRT/OpenVINO 的
> `libnvinfer_builder_resource*.so` 带 executable-stack 标记时，新内核需用
> `patchelf --clear-execstack` 清除后方可加载。

## 构建

```bash
make                 # 编译主程序 bin/trt-infer（含 OpenVINO 后端，若启用）
make quant-eval      # 编译量化评估工具 bin/trt-quant-eval
make volume-seg      # 编译体数据分割工具 bin/trt-volume-seg
make liver3d         # 编译 post3d CUDA 3D 后处理流水线 bin/liver3d
make clean           # 清理
```

构建开关位于 `config/Makefile.config`：`DEBUG`、`SHOW_WARNING`、`ENABLE_OPENVINO`、`APP`。

## 目录结构

```
InferDeploy
├── bin/                    # 可执行文件：trt-infer / trt-quant-eval / trt-volume-seg / liver3d
├── config/                 # Makefile.config（依赖路径、GPU 架构、编译开关）
├── deps/                   # TensorRT-10.16 / OpenVINO / pyenv（Python 依赖环境）
├── include/
│   ├── common/             # ModelParam（Params/枚举/Dims）、ModelCfg
│   ├── core/               # InferBase、TrtBackend、OvBackend、Calibrator
│   ├── tasks/              # Worker、YoloClassifier/YoloDetector/YoloMultitask/UnetSegmenter、TaskUtils
│   └── utils/              # Logger、Timer、Process（CPU/GPU 预处理）、FileUtil、CudaCheck
├── src/                    # 与 include 对应的 .cpp/.cu 实现
├── samples/                # sample.cpp、quant_eval.cpp、volume_seg.cpp
├── post3d/                 # CUDA 3D 后处理子模块（CCL/形态学/高斯，独立 nvcc 构建）
├── data/
│   ├── models/             # 导出/量化/准备脚本（export_yolo.py、quantize_yolo.py、prepare_volume.py 等）与 onnx/engine
│   └── source/             # 测试图像、CT/MR 切片、体数据
├── calibration/            # INT8 校准数据列表
└── docs/                   # 量化原理（量化.md）、目标检测部署实测、3D 肝脏分割部署实测、插图生成脚本（tools/）
```

## 快速开始

```bash
cd ./bin

# 多任务（检测+分割），TensorRT 后端
./trt-infer ../data/models/onnx/yolov8n-seg.onnx ../data/source/crowd.jpg

# 切换 OpenVINO 后端
./trt-infer ../data/models/onnx/yolov8n-seg.onnx ../data/source/crowd.jpg ov

# 肝脏 CT 分割（2D UNet，第 4 参数 seg 指定分割任务）
./trt-infer ../data/models/onnx/liver-unet.onnx ../data/source/ct_liver/lits_v280.tif_z280.png trt seg

# 体数据逐切面分割 + 3D 后处理
./trt-volume-seg ../data/models/onnx/liver-unet.onnx \
    ../data/source/ct_liver/kaggle_seq128 ../data/source/result/kaggle_seq128 \
    trt --post3d --keep-largest

# FP32/INT8 量化精度评估
./trt-quant-eval multi ../data/models/onnx/yolov8n-seg.onnx ../data/source/image_list.txt
```

> 注意：工具默认按 `bin/` 为工作目录解析相对路径（`../data`、`../calibration`、`../engine`、`../result`）。

## 整体架构

### 架构总览

```mermaid
flowchart TD
    A["Main 函数 / 业务调用"] --> B["统一接口层 Worker<br>创建后端 + 按任务分发"]

    B --> C1["分类<br>YoloClassifier"]
    B --> C2["检测<br>YoloDetector"]
    B --> C3["检测+分割多任务<br>YoloMultitask"]
    B --> C4["肝脏 CT 分割<br>UnetSegmenter"]

    C1 & C2 & C3 & C4 --> D["抽象基类 InferBase<br>通用流程 + I/O 缓冲管理"]
    D -->|"组合持有"| E{"InferBackend 接口<br>LoadModel / BindBuffers / Enqueue / Release"}

    E --> F1["TrtBackend<br>TensorRT 引擎构建/推理"]
    E --> F2["OvBackend<br>OpenVINO 编译/推理"]

    D --> G1["分级日志 Logger"]
    D --> G2["双计时器 Timer"]
    D --> G3["预处理 Process<br>CPU(OpenCV) / GPU(CUDA 核函数)"]

    F1 --> H1["TensorRT + CUDA"]
    F2 --> H2["OpenVINO (CPU)"]

    C4 --> I["trt-volume-seg 体数据工具<br>逐切面推理 → 3D 重建"]
    I --> J["post3d CUDA 3D 后处理<br>CCL → 形态学 → 高斯平滑"]
```

### 继承与组合关系

- **任务类**（Classifier/Detector/Multitask/UnetSegmenter）继承 **InferBase** 抽象基类（后端无关），通过组合持有具体后端
- **TrtBackend / OvBackend** 实现 **InferBackend** 接口
- **Logger** 继承 TensorRT 的 **ILogger** 接口

```mermaid
classDiagram
    class InferBase {
        <<abstract>>
        +LoadImage()
        +InitModel()
        +Inference()
    }
    class InferBackend {
        <<interface>>
        +LoadModel()*
        +BindBuffers()*
        +Enqueue()*
        +Release()*
    }
    YoloClassifier --|> InferBase
    YoloDetector --|> InferBase
    YoloMultitask --|> InferBase
    UnetSegmenter --|> InferBase
    InferBase o-- InferBackend : 组合（多态后端）
    TrtBackend --|> InferBackend
    OvBackend --|> InferBackend
    Logger --|> ILogger
```

### 组合关系

- **Worker** 按 `Params.m_eBackend` 创建后端实例，按 `Params.m_eTask` 创建对应任务对象
- **InferBase** 持有 Params、Logger、Timer、CUDA stream、I/O 缓冲与 `IoMeta` 元信息
- **TrtBackend** 持有 Runtime/Engine/Context 等 TensorRT 运行时组件
- **Detector / Multitask / UnetSegmenter** 持有检测结果（bbox/mask）对象

```mermaid
classDiagram
    Worker "1" --* "1" Params : 组合
    Worker "1" --* "0..1" YoloClassifier : 组合
    Worker "1" --* "0..1" YoloDetector : 组合
    Worker "1" --* "0..1" YoloMultitask : 组合
    Worker "1" --* "0..1" UnetSegmenter : 组合

    InferBase "1" --* "1" Params : 组合
    InferBase "1" --* "1" Logger : 组合
    InferBase "1" --* "1" Timer : 组合
    InferBase "1" --* "1" InferBackend : 组合
    TrtBackend "1" --* "1" ICudaEngine : 组合
    TrtBackend "1" --* "1" IExecutionContext : 组合
    OvBackend "1" --* "1" ovCompiledModel : 组合
    OvBackend "1" --* "1" ovInferRequest : 组合

    YoloDetector "1" --* "*" Bbox : 组合
    YoloMultitask "1" --* "*" Bbox : 组合
```

## 通用执行流（InferBase）

所有任务共享同一执行流，后端与任务差异分别由 `InferBackend` 与任务类重写点承载：

```mermaid
flowchart LR
    A["CreateWorker"] --> B["InitModel 首次"]
    B --> B1["Backend.LoadModel<br>构建/加载 engine 或编译模型"]
    B1 --> B2["任务类 SetupIoBuffers<br>按 IoMeta 分配 host/device 缓冲"]
    B2 --> B3["Backend.BindBuffers<br>张量地址绑定"]
    B3 --> C["LoadImage"]
    C --> D["Inference"]
    D --> D1["PreprocessCpu / PreprocessGpu"]
    D1 --> D2["Backend.Enqueue 推理"]
    D2 --> D3["PostprocessCpu / PostprocessGpu"]
    D3 --> E{"继续推理?"}
    E -->|是| C
    E -->|否| F["析构: ReleaseExtraBuffers<br>CleanupBuffers + Backend.Release"]
```

## 各任务流程图

### 分类任务（YoloClassifier）

```mermaid
flowchart LR
    A["读取图像"] --> B["CPU: cv::resize<br/>+ mean/std 归一化 + CHW"]
    A --> C["GPU: ResizeGpu<br/>GPU_BILINEAR + ImageNet mean/std"]
    B --> D["H2D 异步拷贝"]
    C --> E["推理<br/>1x1000 概率向量"]
    D --> E
    E --> F["D2H 回 host"]
    F --> G["argmax 取 top-1"]
    G --> H["输出 ImageNet 标签 + 置信度"]
```

### 检测任务（YoloDetector，YOLOv8）

```mermaid
flowchart LR
    A["读取图像"] --> B["letterbox 预处理<br/>CPU: LetterboxCpu / GPU: WARP_AFFINE<br/>等比缩放 + 黑边清零 + /255"]
    B --> C["推理<br/>输出 1x8400x84 box-major"]
    C --> D["D2H + 解码<br/>cx,cy,w,h → 角点<br/>仿射逆变换回原图坐标<br/>conf > 0.25 过滤"]
    D --> E["同类别贪心 NMS<br/>IoU > 0.45 抑制"]
    E --> F["绘制 bbox + 类别标签<br/>保存结果图"]
```

### 多任务（YoloMultitask，YOLOv8-seg 检测+分割）

```mermaid
flowchart LR
    A["letterbox 预处理"] --> B["双输出推理"]
    B --> B1["检测头 1x8400x116<br/>4坐标+80类+32MC"]
    B --> B2["proto 头 1x32x160x160"]
    B1 --> C["解码 + NMS<br/>得到最终框与 mask 系数 MC"]
    C --> D["MC(N x 32) 与 proto 矩阵乘<br/>reshape 160x160xN"]
    B2 --> D
    D --> E["逐通道 sigmoid<br/>letterbox 有效区裁剪 ROI"]
    E --> F["上采样回原图<br/>> 0.5 二值化"]
    F --> G["mask 半透明叠加 + bbox 绘制<br/>保存结果图"]
```

### 肝脏 CT 分割（UnetSegmenter，2D UNet）

```mermaid
flowchart LR
    A["读取 CT 切片<br/>IMREAD_GRAYSCALE"] --> B["CPU: cv::resize + /255<br/>GPU: ResizeGrayGpu 双线性核函数<br/>训练域: 肝窗 [-100,300]HU"]
    B --> C["推理 1x1x256x256<br/>输出 logits（图内无 Sigmoid）"]
    C --> D["logits > 0.5 二值化<br/>≈ sigmoid 0.62，生成 0/255 mask"]
    D --> E["mask 上采样回原图尺寸<br/>统计肝区面积占比"]
    E --> F["红色半透明叠加可视化<br/>保存结果图 + 二值 mask"]
```

### 体数据逐切面分割 + CUDA 3D 后处理（trt-volume-seg + post3d）

```mermaid
flowchart TD
    A["完整 CT volume<br/>nii / 切片序列"] --> B["prepare_volume.py<br/>HU 窗/直方图匹配 → png 序列"]
    B --> C["trt-volume-seg<br/>逐切面 UnetSegmenter 推理"]
    C --> D["输出: mask_volume.raw 3D 体<br/>逐层 mask png + summary.txt 面积统计"]
    D --> E["post3d_in: slice_%04d.u8 + meta.txt<br/>(W H D dx dy dz)"]
    E --> F["liver3d CUDA 3D 后处理流水线"]
    subgraph F_detail ["post3d 流水线（全程 GPU）"]
        F1["3D 重建: H2D + 二值化"] --> F2["3D CCL 26 连通 union-find<br/>小连通域过滤（去跨层假阳性）"]
        F2 --> F3["3D 形态学开闭（可分离 box）<br/>弥合层间不连续 / 去毛刺"]
        F3 --> F4["3D 高斯平滑 + 再二值化<br/>抑制边缘噪声"]
        F4 --> F5["体积统计 cm³ + PGM 预览"]
    end
    F --> F_detail
    F_detail --> G["liver_final.u8（可导入 3D Slicer/ITK-SNAP）"]
    G --> H["final_slices/final_slice_*.png<br/>后处理结果回导出为逐层 2D 切面"]
```

### INT8 量化与精度评估（trt-quant-eval）

```mermaid
flowchart LR
    A["export_yolo.py<br/>导出 ONNX + box-major 转置"] --> B["quantize_yolo.py<br/>modelopt PTQ 插入 Q/DQ 节点<br/>→ model-int8.onnx"]
    B --> C["BuildEngine 自动检测 Q/DQ<br/>跳过 legacy calibrator<br/>（TRT 10+ 显式量化路径）"]
    C --> D["trt-quant-eval<br/>同一组图跑 FP32 基准 与 INT8"]
    D --> E["指标: 匹配率 / 匹配框平均 IoU / 置信度差<br/>验收线: ≥95% / ≥0.90 / ≤0.05"]
    E --> F{"PASS / FAIL"}
```

## 模型框架介绍

### 用户调用接口

```c++
// 指定参数：模型文件路径、日志等级、模型构建参数（含后端与任务类型）
auto p_worker = CreateWorker(str_onnx_path, e_level, c_params);

// 指定待推理图片, 进行推理
p_worker->Inference(str_img_path);
```

### Worker 类，提供外部调用接口

- 作用：保证代码的可读性，只对用户可见必要接口
- 提供接口：
  - 构造函数：按 `m_eBackend` 创建推理后端（TrtBackend/OvBackend），按 `m_eTask` 创建对应任务对象
  - 执行推理：`InitModel → LoadImage → Inference` 三步编排

```mermaid
classDiagram
class Worker {
    +m_cParams : Params
    +m_pYoloClassifier : unique_ptr~YoloClassifier~
    +m_pYoloDetector : unique_ptr~YoloDetector~
    +m_pYoloMultitask : unique_ptr~YoloMultitask~
    +m_pUnetSegmenter : unique_ptr~UnetSegmenter~
    +Worker(strOnnxPath, cLevel, cParams)
    +Inference(strImagePath)
}
```

### InferBase 抽象基类，核心框架

- 作用：
  - 后端无关的推理基类，封装通用执行流与 I/O 缓冲管理
  - 提供预处理、后处理、缓冲分配接口，具体任务类继承后重写
- 提供接口：
  - `LoadImage`：记录图像路径与推理信息
  - `InitModel`：`LoadModel → SetupIoBuffers → BindBuffers`（首次）/ `ResetTask`（复用）
  - `Inference`：`Preprocess → Enqueue → Postprocess`
  - `GetPrec`：精度后缀（engine 命名）
  - `Get*HostPtr / Get*DevicePtr`：缓冲地址访问器（多输出任务类可重写）

```mermaid
classDiagram
class InferBase {
    <<abstract>>
    +m_strOnnxPath : string
    +m_strImagePath : string
    +m_strOutputPath : string
    +m_matInputImage : cv::Mat
    +m_cParams : Params
    +m_pInputMemory : float_ptr_array
    +m_pOutputMemory : float_ptr_array
    +m_vecInputs : vector~IoMeta~
    +m_vecOutputs : vector~IoMeta~
    +m_cuStream : cudaStream_t
    +m_cLogger : Logger
    +m_cTimer : Timer
    +m_pBackend : shared_ptr~InferBackend~
    +LoadImage(strImagePath)
    +InitModel()
    +Inference()
    +GetPrec(ePrec) string
    +SetupIoBuffers()*
    +ResetTask()*
    +PreprocessCpu()*
    +PreprocessGpu()*
    +PostprocessCpu()*
    +PostprocessGpu()
    +GetInputHostPtr(i)
    +GetInputDevicePtr(i)
    +GetOutputHostPtr(i)
    +GetOutputDevicePtr(i)
}
```

### InferBackend 后端接口与实现

```mermaid
classDiagram
class InferBackend {
    <<interface>>
    +LoadModel(pOwner) bool
    +BindBuffers(pOwner) bool
    +Enqueue(pOwner) bool
    +Release() void
}
class TrtBackend {
    -m_pRuntime : TrtUniquePtr
    -m_pEngine : TrtUniquePtr
    -m_pContext : TrtUniquePtr
    -m_strEnginePath : string
    -BuildEngine() bool
    -LoadEngine() bool
    -SetupRuntime(pData, iSize)
    -SavePlan(nvPlan)
    -PrintNetwork(nvNetwork, bOptimized)
}
class OvBackend {
    -m_pModel : shared_ptr~ov::Model~
    -m_compiledModel : ov::CompiledModel
    -m_inferRequest : ov::InferRequest
}
InferBackend <|-- TrtBackend
InferBackend <|-- OvBackend
```

- **TrtBackend**：`.engine` 存在则直接反序列化加载；否则从 `.onnx` 构建
  （`builder → network → config → parser → serialize`），FP16/INT8 精度、
  Q/DQ 显式量化检测、legacy calibrator、FP32 敏感层回退均在此实现
- **OvBackend**：`read_model → compile_model("CPU") → InferRequest`，
  `ov::Tensor` 零拷贝包裹 host 缓冲；`#ifdef ENABLE_OPENVINO` 条件编译

### 任务类

| 类 | 任务 | 重写要点 |
| --- | --- | --- |
| YoloClassifier | 分类 | ImageNet mean/std 归一化；argmax top-1；`GetResult()` |
| YoloDetector | 检测 | letterbox 预处理；box-major 解码 + NMS；`GetFinalBboxes()` |
| YoloMultitask | 检测+分割 | 双输出缓冲（重写 `GetOutput*Ptr`）；MC×proto mask 生成 |
| UnetSegmenter | 肝脏分割 | 灰度 /255；输出 logits，`>0.5` 二值化；`GetMask()`；`SetSaveDebug()` 体数据模式 |

新增任务仅需继承 `InferBase` 实现 `SetupIoBuffers / Preprocess* / Postprocess*`，
并在 `Worker` 注册任务分支，无需改动任何后端代码。

### Logger 类，继承自 `nvinfer1::ILogger`，日志记录

- 继承 `nvinfer1::ILogger`，重写 `log` 函数
- 支持 FATAL、ERROR、WARN、INFO、VERB、DEBUG 六个级别，颜色编码区分输出

### Timer 类，耗时测量

- C++11 chrono 进行 CPU 计时，CUDA event 进行 GPU 计时
- 支持纳秒、微秒、毫秒、秒多种时间单位，关键节点：preprocess / inference / postprocess

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

### 设备与后端类别

```c++
enum Device { CPU, GPU };

enum InferBackendType {
    TENSORRT,   // NVIDIA TensorRT（GPU）
    OPENVINO    // Intel OpenVINO（CPU，需 ENABLE_OPENVINO=1）
};
```

### 量化精度与校准算法

```c++
enum Precision { FP32, FP16, INT8 };

enum CalibAlgo {
    ENTROPY,  // IInt8EntropyCalibrator2（legacy，仅 TRT 8.x 可用）
    MINMAX    // IInt8MinMaxCalibrator（legacy）
};
// TRT 10+ INT8 走 modelopt 显式量化（Q/DQ ONNX），无需 calibrator
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
    FATAL = 0, ERROR = 1, WARN = 2, INFO = 3, VERB = 4, DEBUG = 5
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
    int                  m_iCalibBatchSize      = 64;    // INT8 校准 batch size
    std::string          m_strCalibListPath;              // 校准数据列表路径
    CalibAlgo            m_eCalibAlgo           = ENTROPY;// INT8 校准算法（legacy 路径）
    InferBackendType     m_eBackend             = TENSORRT; // 推理后端
    std::vector<std::string> m_vecFp32Layers;             // 强制 FP32 的精度敏感层
};
```

## 工程化

- **代码可读性**：`Worker` 对外仅暴露 `创建实例 → 加载图片 → 执行推理` 三步，屏蔽引擎构建、内存管理、CUDA 调度细节；内部按职责拆分（后端/任务/工具），职责边界清晰
- **代码可复用性**：`InferBase` 提炼部署全流程通用范式（模型构建、推理执行、host/device 内存管理、前后处理执行流）；后端由 `InferBackend` 接口统一抽象，任务代码双后端零改动复用
- **高可扩展性**：新增任务仅需继承 `InferBase` 重写前后处理；新增后端仅需实现 `InferBackend` 四方法——核心框架与具体任务/后端完全解耦，最小化改动扩展
- **内存与调用安全性**：`unique_ptr + 自定义 deleter`（TrtUniquePtr）与 `shared_ptr` 管理 TensorRT 运行时对象与后端生命周期；RAII 将资源申请绑定构造函数、释放绑定析构函数（`CleanupBuffers / Release`），从机制上规避泄漏
- **可调试性**：分级日志 `Logger` 六级可控；模型构建、引擎加载、前后处理、推理执行关键节点埋点；`Timer` CPU/GPU 双计时；`PrintNetwork` 输出优化前后网络结构与层精度，快速定位 NMS 输出、数据布局（NCHW/NHWC）等问题

## 示例：肝脏 CT 分割与体数据后处理

基于开源 **LiTS**（Liver Tumor Segmentation Challenge）真实 CT 数据训练的 2D UNet，
通过本框架 TensorRT / OpenVINO 双后端推理（留出切片 Dice ≈ 0.89～0.90）：

```bash
# 单切片推理
./trt-infer ../data/models/onnx/liver-unet.onnx ../data/source/ct_liver/lits_v280.tif_z280.png trt seg

# 体数据逐切面分割 + 3D 后处理（CCL 清除跨层假阳性、形态学弥合、高斯平滑）
./trt-volume-seg ../data/models/onnx/liver-unet.onnx \
    ../data/source/ct_liver/kaggle_seq128 ../data/source/result/kaggle_seq128 \
    trt --post3d --keep-largest
# 产物: mask_volume.raw / post3d_out/liver_final.u8 / final_slices/*.png / 体积统计
```

> 精度边界：模型在 LiTS 域（同窗宽/增强协议）内可靠；跨中心/跨协议数据存在域偏移，
> 需域适配后使用。详见 `docs/量化.md`。

## 参考

- 韩君，《CUDA与TensorRT部署实战课程》
- [kalfazed/tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)：框架架构与量化章节参考
- [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)：ONNX 显式量化（Q/DQ）
