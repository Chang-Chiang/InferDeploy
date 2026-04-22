#include "include/core/InferTRT.h"
#include "include/utils/ModelUtil.h"

InferTRT::InferTRT(string strOnnxPath, Level cLevel, Params cParams) {
    m_strOnnxPath    = strOnnxPath;
    m_iWorkspaceSize = WORKSPACESIZE;
    m_pLogger        = make_shared<Logger>(cLevel);
    m_pTimer         = make_shared<Timer>();
    m_pParams        = new Params(cParams);
    m_strEnginePath  = changePath(strOnnxPath, "../engine", ".engine", getPrec(params.prec));
}

void InferTRT::LoadImage(string strImagePath) {
    if (!fileExists(strImagePath)){
        LOGE("%s not found", strImagePath.c_str());
    }
    else {
        m_strImagePath = strImagePath;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", getFileName(m_strOnnxPath).c_str());
        LOG("\tImage:      %s", getFileName(m_strImagePath).c_str());
        LOG("\tPrecision:  %s", getPrec(m_pParams->prec).c_str());
    }
}

void InferTRT::InitModel() {
    if (m_pContext == nullptr) {
        if (!fileExists(m_strEnginePath)) {
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
    auto p_builder = shared_ptr<IBuilder>(createInferBuilder(*m_pLogger), DestroyTrtPtr<IBuilder>);
    auto p_network = shared_ptr<INetworkDefinition>(p_builder->createNetworkV2(1), DestroyTrtPtr<INetworkDefinition>);
    auto p_config  = shared_ptr<IBuilderConfig>(p_builder->createBuilderConfig(), DestroyTrtPtr<IBuilderConfig>);
    auto p_parser  = shared_ptr<IParser>(createParser(*p_network, *m_pLogger), DestroyTrtPtr<IParser>);

    p_config->setMaxWorkspaceSize(m_workspaceSize);
    p_config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED); //这里也可以设置为kDETAIL;

    if (!p_parser->parseFromFile(m_strOnnxPath.c_str(), 1)) {
        return false;
    }

    if (p_builder->platformHasFastFp16() && m_params->prec == FP16) {
        p_config->setFlag(BuilderFlag::kFP16);
        p_config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }
    else if (p_builder->platformHasFastInt8() && m_params->prec == INT8) {
        p_config->setFlag(BuilderFlag::kINT8);
        p_config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    shared_ptr<Int8EntropyCalibrator> p_calibrator(
        new Int8EntropyCalibrator(
            4, 
            "calibration/calibration_list_coco.txt", 
            "calibration/calibration_table.txt",
            3 * 224 * 224, 224, 224));

    p_config->setInt8Calibrator(p_calibrator.get());

    auto p_engine  = shared_ptr<ICudaEngine>(p_builder->buildEngineWithConfig(*p_network, *p_config), DestroyTrtPtr<ICudaEngine>);
    auto plan      = p_builder->buildSerializedNetwork(*p_network, *p_config);
    auto p_runtime = shared_ptr<IRuntime>(createInferRuntime(*m_pLogger), DestroyTrtPtr<IRuntime>);

    SavePlan(*plan);

    Setup(plan->data(), plan->size());

    LOGV("Before TensorRT optimization");
    PrintNetwork(*p_network, false);
    LOGV("After TensorRT optimization");
    PrintNetwork(*p_network, true);

    return true;
}

bool InferTRT::LoadEngine() {

    if (!fileExists(m_strEnginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    std::vector<unsigned char> vec_model_data;
    vec_model_data     = LoadFile(m_strEnginePath);

    Setup(vec_model_data.data(), vec_model_data.size());

    return true;
}

void InferTRT::SavePlan(IHostMemory& plan) {
    auto f = fopen(m_strEnginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}

void InferTRT::Inference() {
    if (m_params->dev == CPU) {
        PreprocessCpu();
    }
    else {
        PreprocessGpu();
    }

    EnqueueBindings();

    if (m_params->dev == CPU) {
        PostprocessCpu();
    }
    else {
        PostprocessGpu();
    }
}

bool InferTRT::EnqueueBindings() {
    m_pTimer->StartGpu();
    if (!m_pContext->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_pTimer->StopGpu("trt-inference(GPU)");
    return true;
}

void InferTRT::PrintNetwork(INetworkDefinition &nvNetwork, bool bOptimized) {

    int i_input_cnt = nvNetwork.getNbInputs();
    int i_output_cnt = nvNetwork.getNbOutputs();
    string str_layer_info;

    for (int i = 0; i < i_input_cnt; i++) {
        auto input = nvNetwork.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), PrintTensorShape(input).c_str());
    }

    for (int i = 0; i < i_output_cnt; i++) {
        auto output = nvNetwork.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), PrintTensorShape(output).c_str());
    }
    
    int i_layer_cnt = bOptimized ? m_engine->getNbLayers() : nvNetwork.getNbLayers();
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
        auto p_inspector = shared_ptr<IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < i_layer_cnt; i++) {
            LOGV("layer_info: %s", p_inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

string InferTRT::GetPrec(Precision cPrec) {
    switch(cPrec) {
        case Precision::FP16:   return "fp16";
        case Precision::INT8:   return "int8";
        default:                return "fp32";
    }
}