#include "include/core/InferTRT.h"
#include "include/core/Calibrator.h"
#include "include/utils/ModelUtil.h"
#include "include/utils/FileUtil.h"

#include <NvOnnxParser.h>
#include <unistd.h>

InferTRT::InferTRT(std::string strOnnxPath, Level cLevel, Params cParams) {
    m_strOnnxPath    = strOnnxPath;
    m_iWorkspaceSize = WORKSPACESIZE;
    m_pLogger        = std::make_shared<Logger>(cLevel);
    m_pTimer         = std::make_shared<Timer>();
    m_pParams        = new Params(cParams);
    m_strEnginePath  = ChangePath(strOnnxPath, "../engine", ".engine", GetPrec(m_pParams->m_ePrec));
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
        LOG("\tPrecision:  %s", GetPrec(m_pParams->m_ePrec).c_str());
    }
}

void InferTRT::InitModel() {
    if (m_pContext == nullptr) {
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
        m_pTimer->Init();
        ResetTask();
    }
}

bool InferTRT::BuildEngine() {
    auto p_builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_pLogger), DestroyTrtPtr<nvinfer1::IBuilder>);
    if (!p_builder) {
        LOGE("Failed to create TensorRT builder");
        return false;
    }

    auto p_network = std::shared_ptr<nvinfer1::INetworkDefinition>(p_builder->createNetworkV2(1), DestroyTrtPtr<nvinfer1::INetworkDefinition>);
    if (!p_network) {
        LOGE("Failed to create TensorRT network");
        return false;
    }

    auto p_config  = std::shared_ptr<nvinfer1::IBuilderConfig>(p_builder->createBuilderConfig(), DestroyTrtPtr<nvinfer1::IBuilderConfig>);
    if (!p_config) {
        LOGE("Failed to create TensorRT builder config");
        return false;
    }

    auto p_parser  = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*p_network, *m_pLogger), DestroyTrtPtr<nvonnxparser::IParser>);
    if (!p_parser) {
        LOGE("Failed to create ONNX parser");
        return false;
    }

    p_config->setMaxWorkspaceSize(m_iWorkspaceSize);
    p_config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    if (!p_parser->parseFromFile(m_strOnnxPath.c_str(), 1)) {
        LOGE("Failed to parse ONNX file: %s", m_strOnnxPath.c_str());
        // Count errors in parsing
        int nerror = p_parser->getNbErrors();
        for (int i = 0; i < nerror; ++i) {
            auto error = p_parser->getError(i);
            LOGE("ONNX Parser Error: %s", error->desc());
        }
        return false;
    }

    // Auto-detect input dimensions from parsed network
    auto input = p_network->getInput(0);
    auto dims = input->getDimensions();
    int iInputC = dims.d[0];  // channels
    int iInputH = dims.d[1];  // height
    int iInputW = dims.d[2];  // width

    // Update Params with auto-detected dimensions
    m_pParams->m_cImgInfo = {iInputH, iInputW, iInputC};
    LOG("Auto-detected input shape: %d x %d x %d", iInputC, iInputH, iInputW);

    if (p_builder->platformHasFastFp16() && m_pParams->m_ePrec == FP16) {
        p_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    else if (p_builder->platformHasFastInt8() && m_pParams->m_ePrec == INT8) {
        p_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

        // Use auto-detected dimensions from Params
        int iInputC = m_pParams->m_cImgInfo.m_iChannel;
        int iInputH = m_pParams->m_cImgInfo.m_iHeight;
        int iInputW = m_pParams->m_cImgInfo.m_iWidth;

        std::shared_ptr<Int8EntropyCalibrator> p_calibrator(
            new Int8EntropyCalibrator(
                4,
                "../calibration/calibration_list.txt",
                "../calibration_table.txt",
                iInputC * iInputH * iInputW, iInputW, iInputH));

        p_config->setInt8Calibrator(p_calibrator.get());
    }

    auto p_engine  = std::shared_ptr<nvinfer1::ICudaEngine>(p_builder->buildEngineWithConfig(*p_network, *p_config), DestroyTrtPtr<nvinfer1::ICudaEngine>);
    if (!p_engine) {
        LOGE("Failed to build TensorRT engine");
        return false;
    }

    auto plan = p_engine->serialize();
    if (!plan) {
        LOGE("Failed to serialize TensorRT engine");
        return false;
    }

    auto p_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_pLogger), DestroyTrtPtr<nvinfer1::IRuntime>);
    if (!p_runtime) {
        LOGE("Failed to create TensorRT runtime");
        return false;
    }

    SavePlan(*plan);

    Setup(plan->data(), plan->size());

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
    // Resolve absolute path to avoid relative path issues
    std::string strAbsPath = m_strEnginePath;
    if (m_strEnginePath.find("../") == 0) {
        // Convert relative path to absolute
        char szCurrDir[1024];
        if (getcwd(szCurrDir, sizeof(szCurrDir)) != nullptr) {
            strAbsPath = std::string(szCurrDir) + "/" + m_strEnginePath;
        }
    }

    // Extract directory from absolute path
    std::string strDir = strAbsPath.substr(0, strAbsPath.find_last_of("/"));
    if (!IsFileExist(strDir)) {
        LOG("Creating engine directory: %s", strDir.c_str());
        if (!CreateDir(strDir)) {
            LOGE("Failed to create engine directory: %s", strDir.c_str());
            return;
        }
    }

    auto f = fopen(strAbsPath.c_str(), "wb");
    if (!f) {
        LOGE("Failed to open engine file for writing: %s", strAbsPath.c_str());
        return;
    }
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
    LOG("Engine saved to: %s", strAbsPath.c_str());
}

void InferTRT::Inference() {
    if (m_pParams->m_eDevice == CPU) {
        PreprocessCpu();
    }
    else {
        PreprocessGpu();
    }

    EnqueueBindings();

    if (m_pParams->m_eDevice == CPU) {
        PostprocessCpu();
    }
    else {
        PostprocessGpu();
    }
}

bool InferTRT::EnqueueBindings() {
    m_pTimer->StartGpu();
    if (!m_pContext->enqueueV2((void**)m_pBindings, m_cuStream, nullptr)) {
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_pTimer->StopGpu("trt-inference(GPU)");
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
        auto p_inspector = std::shared_ptr<nvinfer1::IEngineInspector>(m_pEngine->createEngineInspector());
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