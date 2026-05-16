#include "include/core/InferTRT.h"
#include "include/core/Calibrator.h"
#include "include/utils/ModelUtil.h"
#include "include/utils/FileUtil.h"

#include <NvOnnxParser.h>
#include <unistd.h>

InferTRT::InferTRT(std::string strOnnxPath, Level cLevel, Params cParams)
    : m_cLogger(cLevel)
    , m_cParams(cParams)
    , m_strOnnxPath(strOnnxPath)
    , m_iWorkspaceSize(WORKSPACESIZE) {
    m_strEnginePath  = ChangePath(strOnnxPath, "../engine", ".engine", GetPrec(m_cParams.m_ePrec));
}

void InferTRT::LoadImage(std::string strImagePath) {
    if (!IsFileExist(strImagePath)) {
        LOGE("%s not found", strImagePath.c_str());
    }
    else {
        m_strImagePath = strImagePath;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", GetFileName(m_strOnnxPath).c_str());
        LOG("\tImage:      %s", GetFileName(m_strImagePath).c_str());
        LOG("\tPrecision:  %s", GetPrec(m_cParams.m_ePrec).c_str());
    }
}

void InferTRT::InitModel() {
    if (!m_pContext) {
        if (!IsFileExist(m_strEnginePath)) {
            LOG("%s not found. Building trt engine...", m_strEnginePath.c_str());
            BuildEngine();
        }
        else {
            LOG("%s has been generated! loading trt engine...", m_strEnginePath.c_str());
            LoadEngine();
        }
    }
    else {
        m_cTimer.Init();
        ResetTask();
    }
}

bool InferTRT::BuildEngine() {
    TrtUniquePtr<nvinfer1::IBuilder> p_builder(nvinfer1::createInferBuilder(m_cLogger));
    if (!p_builder) {
        LOGE("Failed to create TensorRT builder");
        return false;
    }

    TrtUniquePtr<nvinfer1::INetworkDefinition> p_network(p_builder->createNetworkV2(1));
    if (!p_network) {
        LOGE("Failed to create TensorRT network");
        return false;
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> p_config(p_builder->createBuilderConfig());
    if (!p_config) {
        LOGE("Failed to create TensorRT builder config");
        return false;
    }

    TrtUniquePtr<nvonnxparser::IParser> p_parser(nvonnxparser::createParser(*p_network, m_cLogger));
    if (!p_parser) {
        LOGE("Failed to create ONNX parser");
        return false;
    }

    p_config->setMaxWorkspaceSize(m_iWorkspaceSize);
    p_config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!p_parser->parseFromFile(m_strOnnxPath.c_str(), 1)) {

        LOGE("Failed to parse ONNX file: %s", m_strOnnxPath.c_str());

        int nerror = p_parser->getNbErrors();
        for (int i = 0; i < nerror; ++i) {
            auto error = p_parser->getError(i);
            LOGE("ONNX Parser Error: %s", error->desc());
        }
        return false;
    }

    // 根据模型读取输入模型的图像信息，并更新 Params 结构体中的 m_cImgInfo 成员变量
    auto input = p_network->getInput(0);
    auto dims = input->getDimensions();
    int i_input_c = dims.d[0];
    int i_input_h = dims.d[1];
    int i_input_w = dims.d[2];
    m_cParams.m_cImgInfo = {i_input_h, i_input_w, i_input_c};
    LOG("Auto-detected input shape: %d x %d x %d", i_input_c, i_input_h, i_input_w);

    if (p_builder->platformHasFastFp16() && m_cParams.m_ePrec == FP16) {
        p_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    else if (p_builder->platformHasFastInt8() && m_cParams.m_ePrec == INT8) {
        p_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

        int i_input_c = m_cParams.m_cImgInfo.m_iChannel;
        int i_input_h = m_cParams.m_cImgInfo.m_iHeight;
        int i_input_w = m_cParams.m_cImgInfo.m_iWidth;

        std::unique_ptr<Int8EntropyCalibrator> p_calibrator(
            std::make_unique<Int8EntropyCalibrator>(
                64,
                "../calibration/calibration_list.txt",
                "../calibration_table.txt",
                i_input_c * i_input_h * i_input_w, i_input_w, i_input_h));

        p_config->setInt8Calibrator(p_calibrator.get());
    }

    // 激活层精度设置为 FP32，避免部分层被优化成 FP16/INT8 导致精度下降
    for (int i = 0; i < p_network->getNbLayers(); i++) {
        auto layer = p_network->getLayer(i);
        if (layer->getName() == "/model.22/proto/cv3/act/Mul") {
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        }
    }

    TrtUniquePtr<nvinfer1::ICudaEngine> p_engine(p_builder->buildEngineWithConfig(*p_network, *p_config));
    if (!p_engine) {
        LOGE("Failed to build TensorRT engine");
        return false;
    }

    auto plan = p_engine->serialize();
    if (!plan) {
        LOGE("Failed to serialize TensorRT engine");
        return false;
    }

    TrtUniquePtr<nvinfer1::IRuntime> p_runtime(nvinfer1::createInferRuntime(m_cLogger));
    if (!p_runtime) {
        LOGE("Failed to create TensorRT runtime");
        return false;
    }

    // 保存序列化后的 engine 文件
    SavePlan(*plan);

    // 加载引擎并创建上下文
    Setup(plan->data(), plan->size());

    // 打印优化前后的网络结构信息
    LOGV("Before TensorRT optimization");
    PrintNetwork(*p_network, false);
    LOGV("After TensorRT optimization");
    PrintNetwork(*p_network, true);

    return true;
}

bool InferTRT::LoadEngine() {

    if (!IsFileExist(m_strEnginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    std::vector<unsigned char> vec_model_data;
    vec_model_data     = LoadFile(m_strEnginePath);

    Setup(vec_model_data.data(), vec_model_data.size());

    return true;
}

void InferTRT::SavePlan(nvinfer1::IHostMemory& plan) {

    std::string str_abs_path = m_strEnginePath;
    if (m_strEnginePath.find("../") == 0) {

        char sz_cur_dir[1024];
        if (getcwd(sz_cur_dir, sizeof(sz_cur_dir)) != nullptr) {
            str_abs_path = std::string(sz_cur_dir) + "/" + m_strEnginePath;
        }
    }

    std::string str_dir = str_abs_path.substr(0, str_abs_path.find_last_of("/"));
    if (!IsFileExist(str_dir)) {
        LOG("Creating engine directory: %s", str_dir.c_str());
        if (!CreateDir(str_dir)) {
            LOGE("Failed to create engine directory: %s", str_dir.c_str());
            return;
        }
    }

    auto f = fopen(str_abs_path.c_str(), "wb");
    if (!f) {
        LOGE("Failed to open engine file for writing: %s", str_abs_path.c_str());
        return;
    }
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
    LOG("Engine saved to: %s", str_abs_path.c_str());
}

void InferTRT::Inference() {
    if (m_cParams.m_eDevice == CPU) {
        PreprocessCpu();
    }
    else {
        PreprocessGpu();
    }

    EnqueueBindings();

    if (m_cParams.m_eDevice == CPU) {
        PostprocessCpu();
    }
    else {
        PostprocessGpu();
    }
}

bool InferTRT::EnqueueBindings() {
    m_cTimer.StartGpu();
    if (!m_pContext->enqueueV2((void**)m_pBindings, m_cuStream, nullptr)) {
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_cTimer.StopGpu("trt-inference(GPU)");
    return true;
}

void InferTRT::PrintNetwork(nvinfer1::INetworkDefinition &nvNetwork, bool bOptimized) {

    int i_input_cnt = nvNetwork.getNbInputs();
    int i_output_cnt = nvNetwork.getNbOutputs();
    std::string str_layer_info;

    for (int i = 0; i < i_input_cnt; i++) {
        auto input = nvNetwork.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), PrintTensorShape(input).c_str());
    }

    for (int i = 0; i < i_output_cnt; i++) {
        auto output = nvNetwork.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), PrintTensorShape(output).c_str());
    }
    
    int i_layer_cnt = bOptimized ? m_pEngine->getNbLayers() : nvNetwork.getNbLayers();
    LOGV("network has %d layers", i_layer_cnt);

    if (!bOptimized) {
        for (int i = 0; i < i_layer_cnt; i++) {
            char ch_layer_info[1000];
            auto layer   = nvNetwork.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                PrintTensorShape(input).c_str(),
                PrintTensorShape(output).c_str(),
                GetPrecision(layer->getPrecision()).c_str());
        }

    }
    else {
        TrtUniquePtr<nvinfer1::IEngineInspector> p_inspector(m_pEngine->createEngineInspector());
        for (int i = 0; i < i_layer_cnt; i++) {
            LOGV("layer_info: %s", p_inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

std::string InferTRT::GetPrec(Precision cPrec) {
    switch(cPrec) {
        case Precision::FP16:   return "fp16";
        case Precision::INT8:   return "int8";
        default:                return "fp32";
    }
}