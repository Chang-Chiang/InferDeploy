#include "include/core/TrtBackend.h"

#include "include/core/Calibrator.h"
#include "include/utils/ModelUtil.h"
#include "include/utils/FileUtil.h"
#include "include/utils/CudaCheck.h"

#include <NvOnnxParser.h>
#include <unistd.h>

bool TrtBackend::LoadModel(InferBase* pOwner) {
    m_strEnginePath =
        ChangePath(pOwner->m_strOnnxPath, "../engine", ".engine", pOwner->GetPrec(pOwner->m_cParams.m_ePrec));

    if (!IsFileExist(m_strEnginePath)) {
        LOG("%s not found. Building trt engine...", m_strEnginePath.c_str());

        TrtUniquePtr<nvinfer1::IBuilder> p_builder(nvinfer1::createInferBuilder(pOwner->m_cLogger));
        if (!p_builder) {
            LOGE("Failed to create TensorRT builder");
            return false;
        }

        // TensorRT 10+ 默认即为 explicit batch 模式，无需再显式指定 flag
        TrtUniquePtr<nvinfer1::INetworkDefinition> p_network(p_builder->createNetworkV2(0));
        if (!p_network) {
            LOGE("Failed to create TensorRT network");
            return false;
        }

        TrtUniquePtr<nvinfer1::IBuilderConfig> p_config(p_builder->createBuilderConfig());
        if (!p_config) {
            LOGE("Failed to create TensorRT builder config");
            return false;
        }

        // TensorRT 10+ 移除了 setMaxWorkspaceSize，改用内存池接口
        p_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                     static_cast<std::size_t>(pOwner->m_cParams.m_iWorkspaceSize));
        p_config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

        TrtUniquePtr<nvonnxparser::IParser> p_parser(nvonnxparser::createParser(*p_network, pOwner->m_cLogger));
        if (!p_parser) {
            LOGE("Failed to create ONNX parser");
            return false;
        }

        if (!p_parser->parseFromFile(pOwner->m_strOnnxPath.c_str(), 1)) {
            LOGE("Failed to parse ONNX file: %s", pOwner->m_strOnnxPath.c_str());

            int nerror = p_parser->getNbErrors();
            for (int i = 0; i < nerror; ++i) {
                auto error = p_parser->getError(i);
                LOGE("ONNX Parser Error: %s", error->desc());
            }
            return false;
        }

        // 根据模型读取输入维度（NCHW 含 batch: [N, C, H, W]），回写 Params.m_cImgInfo
        auto input = p_network->getInput(0);
        auto dims  = input->getDimensions();
        int  i_input_c, i_input_h, i_input_w;
        if (dims.nbDims == 4) {
            i_input_c = dims.d[1];
            i_input_h = dims.d[2];
            i_input_w = dims.d[3];
        }
        else {
            i_input_c = dims.d[0];
            i_input_h = dims.d[1];
            i_input_w = dims.d[2];
        }
        pOwner->m_cParams.m_cImgInfo = {i_input_h, i_input_w, i_input_c};
        LOG("Auto-detected input shape: %d x %d x %d (C x H x W)", i_input_c, i_input_h, i_input_w);

        if (p_builder->platformHasFastFp16() && pOwner->m_cParams.m_ePrec == FP16) {
            p_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        }
        else if (p_builder->platformHasFastInt8() && pOwner->m_cParams.m_ePrec == INT8) {
            // 检测是否为显式量化模型（含 QuantizeLinear/DequantizeLinear 节点，如 modelopt PTQ 产物）
            // 显式量化由 Q/DQ 节点决定精度，无需 kINT8 flag 与 calibrator；
            // implicit calibration 自 TRT 10.1 起弃用，仅保留给 TRT 8.x 旧环境使用
            bool b_explicit_quant = false;
            for (int i = 0; i < p_network->getNbLayers(); i++) {
                if (p_network->getLayer(i)->getType() == nvinfer1::LayerType::kQUANTIZE) {
                    b_explicit_quant = true;
                    break;
                }
            }

            if (b_explicit_quant) {
                // modelopt 产出的 Q/DQ 模型中部分层/常量为 FP16 类型，需要允许 FP16 才能构建
                p_config->setFlag(nvinfer1::BuilderFlag::kFP16);
                LOG("Explicitly quantized model (Q/DQ) detected, building INT8 engine without calibrator...");
            }
            else {
                p_config->setFlag(nvinfer1::BuilderFlag::kINT8);
                p_config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

                // 校准表路径带算法后缀：更换校准算法后 dynamic range 统计信息无法复用，必须重新校准
                std::string str_calib_table = pOwner->m_cParams.m_eCalibAlgo == MINMAX
                                                  ? "../calibration_table_minmax.txt"
                                                  : "../calibration_table_entropy.txt";

                std::unique_ptr<nvinfer1::IInt8Calibrator> p_calibrator;
                if (pOwner->m_cParams.m_eCalibAlgo == MINMAX) {
                    p_calibrator = std::make_unique<Int8MinMaxCalibrator>(
                        pOwner->m_cParams, pOwner->m_cParams.m_strCalibListPath, str_calib_table);
                }
                else {
                    p_calibrator = std::make_unique<Int8EntropyCalibrator>(
                        pOwner->m_cParams, pOwner->m_cParams.m_strCalibListPath, str_calib_table);
                }
                LOG("INT8 calibrator: %s, calibration list: %s",
                    pOwner->m_cParams.m_eCalibAlgo == MINMAX ? "IInt8MinMaxCalibrator" : "IInt8EntropyCalibrator2",
                    pOwner->m_cParams.m_strCalibListPath.c_str());

                p_config->setInt8Calibrator(p_calibrator.get());
            }
        }

        // 将用户指定的层设置为 FP32 精度，避免被优化成 FP16/INT8 导致精度下降
        for (int i = 0; i < p_network->getNbLayers(); i++) {
            auto layer = p_network->getLayer(i);
            for (const auto& fp32_layer : pOwner->m_cParams.m_vecFp32Layers) {
                if (layer->getName() == fp32_layer) {
                    layer->setPrecision(nvinfer1::DataType::kFLOAT);
                    layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
                    break;
                }
            }
        }

        // 动态输入（如导出时启用 dynamic batch）需要设置 optimization profile，此处固定 batch=1
        for (int i = 0; i < p_network->getNbInputs(); i++) {
            auto           in       = p_network->getInput(i);
            nvinfer1::Dims d        = in->getDimensions();
            nvinfer1::Dims dims_min = d, dims_opt = d, dims_max = d;
            bool           b_dynamic = false;
            for (int j = 0; j < d.nbDims; j++) {
                if (d.d[j] == -1) {
                    b_dynamic     = true;
                    dims_min.d[j] = dims_opt.d[j] = dims_max.d[j] = 1;
                }
            }
            if (b_dynamic) {
                auto profile = p_builder->createOptimizationProfile();
                profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMIN, dims_min);
                profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
                profile->setDimensions(in->getName(), nvinfer1::OptProfileSelector::kMAX, dims_max);
                p_config->addOptimizationProfile(profile);
                LOG("Dynamic input detected: %s, optimization profile set to batch=1", in->getName());
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

        // 保存序列化后的 engine 文件
        SavePlan(pOwner, *plan);

        // 加载引擎并创建上下文
        SetupRuntime(pOwner, plan->data(), plan->size());

        // 打印优化前后的网络结构信息
        LOGV("Before TensorRT optimization");
        PrintNetwork(pOwner, *p_network, false);
        LOGV("After TensorRT optimization");
        PrintNetwork(pOwner, *p_network, true);
    }
    else {
        LOG("%s has been generated! loading trt engine...", m_strEnginePath.c_str());
        if (!LoadEngine(pOwner)) {
            return false;
        }
    }

    return m_pContext != nullptr;
}

bool TrtBackend::LoadEngine(InferBase* pOwner) {
    if (!IsFileExist(m_strEnginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    std::vector<unsigned char> vec_model_data = LoadFile(m_strEnginePath);
    SetupRuntime(pOwner, vec_model_data.data(), vec_model_data.size());

    return m_pContext != nullptr;
}

void TrtBackend::SetupRuntime(InferBase* pOwner, void const* pData, std::size_t iSize) {
    m_pRuntime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(pOwner->m_cLogger));
    m_pEngine  = TrtUniquePtr<nvinfer1::ICudaEngine>(m_pRuntime->deserializeCudaEngine(pData, iSize));
    m_pContext = TrtUniquePtr<nvinfer1::IExecutionContext>(m_pEngine->createExecutionContext());

    // TensorRT 10+ 按名称管理 I/O tensor，构建时收集输入/输出张量元信息
    pOwner->m_vecInputs.clear();
    pOwner->m_vecOutputs.clear();
    for (int i = 0; i < m_pEngine->getNbIOTensors(); i++) {
        const char* ch_name = m_pEngine->getIOTensorName(i);
        IoMeta      c_meta;
        c_meta.strName     = ch_name;
        auto nv_dims       = m_pEngine->getTensorShape(ch_name);
        c_meta.dims.nbDims = nv_dims.nbDims;
        for (int j = 0; j < nv_dims.nbDims; j++) {
            c_meta.dims.d[j] = nv_dims.d[j];
        }

        if (m_pEngine->getTensorIOMode(ch_name) == nvinfer1::TensorIOMode::kINPUT) {
            pOwner->m_vecInputs.emplace_back(c_meta);
        }
        else {
            pOwner->m_vecOutputs.emplace_back(c_meta);
        }
    }
}

bool TrtBackend::BindBuffers(InferBase* pOwner) {
    // 输入绑定 device 缓冲；输出按任务类的缓冲访问器绑定（多输出任务各自管理）
    for (int i = 0; i < static_cast<int>(pOwner->m_vecInputs.size()); i++) {
        if (!m_pContext->setTensorAddress(pOwner->m_vecInputs[i].strName.c_str(), pOwner->GetInputDevicePtr(i))) {
            LOGE("Failed to set input tensor address: %s", pOwner->m_vecInputs[i].strName.c_str());
            return false;
        }
    }
    for (int i = 0; i < static_cast<int>(pOwner->m_vecOutputs.size()); i++) {
        if (!m_pContext->setTensorAddress(pOwner->m_vecOutputs[i].strName.c_str(), pOwner->GetOutputDevicePtr(i))) {
            LOGE("Failed to set output tensor address: %s", pOwner->m_vecOutputs[i].strName.c_str());
            return false;
        }
    }
    return true;
}

bool TrtBackend::Enqueue(InferBase* pOwner) {
    pOwner->m_cTimer.StartGpu();
    // TensorRT 10+ 移除 enqueueV2，改用 enqueueV3（地址已在 BindBuffers 中绑定）
    if (!m_pContext->enqueueV3(pOwner->m_cuStream)) {
        return false;
    }
    pOwner->m_cTimer.StopGpu("trt-inference(GPU)");
    return true;
}

void TrtBackend::Release() {
    m_pContext.reset();
    m_pEngine.reset();
    m_pRuntime.reset();
}

void TrtBackend::SavePlan(InferBase* pOwner, nvinfer1::IHostMemory& plan) {

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

void TrtBackend::PrintNetwork(InferBase* pOwner, nvinfer1::INetworkDefinition& nvNetwork, bool bOptimized) {

    int i_input_cnt  = nvNetwork.getNbInputs();
    int i_output_cnt = nvNetwork.getNbOutputs();

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
            auto layer = nvNetwork.getLayer(i);
            auto input = layer->getInput(0);
            if (input == nullptr) {
                continue;
            }
            auto output = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", layer->getName(), PrintTensorShape(input).c_str(),
                 PrintTensorShape(output).c_str(), GetPrecision(layer->getPrecision()).c_str());
        }
    }
    else {
        // IEngineInspector 需先于 engine 释放，用 TrtUniquePtr 管理生命周期
        TrtUniquePtr<nvinfer1::IEngineInspector> p_inspector(m_pEngine->createEngineInspector());
        for (int i = 0; i < i_layer_cnt; i++) {
            LOGV("layer_info: %s", p_inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}
