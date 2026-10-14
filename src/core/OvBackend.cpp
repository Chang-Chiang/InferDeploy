#ifdef ENABLE_OPENVINO

    #include "include/core/OvBackend.h"

    #include "include/utils/FileUtil.h"
    #include "include/utils/CudaCheck.h"

bool OvBackend::LoadModel(InferBase* pOwner) {
    if (!IsFileExist(pOwner->m_strOnnxPath)) {
        LOGE("onnx model not found: %s", pOwner->m_strOnnxPath.c_str());
        return false;
    }

    LOG("OpenVINO: reading model %s...", pOwner->m_strOnnxPath.c_str());
    ov::Core core;
    m_pModel = core.read_model(pOwner->m_strOnnxPath);

    // 输入 shape 回写 Params.m_cImgInfo（NCHW: [N, C, H, W]，动态维度固定为 1）
    // 注意：需在 compile 之前完成回写，任务类 SetupIoBuffers 依赖该信息
    const ov::PartialShape ps_input = m_pModel->input().get_partial_shape();
    auto                   dim_len  = [&ps_input](std::size_t j) {
        return ps_input[j].is_dynamic() ? 1 : static_cast<int>(ps_input[j].get_length());
    };
    int i_input_c, i_input_h, i_input_w;
    if (ps_input.size() == 4) {
        i_input_c = dim_len(1);
        i_input_h = dim_len(2);
        i_input_w = dim_len(3);
    }
    else {
        i_input_c = dim_len(0);
        i_input_h = dim_len(1);
        i_input_w = dim_len(2);
    }
    pOwner->m_cParams.m_cImgInfo = {i_input_h, i_input_w, i_input_c};
    LOG("Auto-detected input shape: %d x %d x %d (C x H x W)", i_input_c, i_input_h, i_input_w);

    // TODO: 支持 GPU/NPU 设备与精度提示（当前 NVIDIA 机器仅 CPU 插件可用）
    LOG("OpenVINO: compiling model for device: CPU...");
    m_compiledModel = core.compile_model(m_pModel, "CPU");
    m_inferRequest  = m_compiledModel.create_infer_request();

    // 收集输入/输出元信息（动态 batch 模型：动态维度固定为 1，与推理时 batch=1 一致）
    pOwner->m_vecInputs.clear();
    pOwner->m_vecOutputs.clear();
    for (std::size_t i = 0; i < m_pModel->inputs().size(); i++) {
        const ov::Output<const ov::Node>& input  = m_pModel->inputs()[i];
        const ov::PartialShape            pshape = input.get_partial_shape();

        IoMeta c_meta;
        c_meta.strName     = input.get_any_name();
        c_meta.dims.nbDims = static_cast<int>(pshape.size());
        for (std::size_t j = 0; j < pshape.size() && j < 8; j++) {
            c_meta.dims.d[j] = pshape[j].is_dynamic() ? 1 : static_cast<int>(pshape[j].get_length());
        }
        pOwner->m_vecInputs.emplace_back(c_meta);
    }
    for (std::size_t i = 0; i < m_pModel->outputs().size(); i++) {
        const ov::Output<const ov::Node>& output = m_pModel->outputs()[i];
        const ov::PartialShape            pshape = output.get_partial_shape();

        IoMeta c_meta;
        c_meta.strName     = output.get_any_name();
        c_meta.dims.nbDims = static_cast<int>(pshape.size());
        for (std::size_t j = 0; j < pshape.size() && j < 8; j++) {
            c_meta.dims.d[j] = pshape[j].is_dynamic() ? 1 : static_cast<int>(pshape[j].get_length());
        }
        pOwner->m_vecOutputs.emplace_back(c_meta);
    }

    LOG("OpenVINO model loaded: %d input(s), %d output(s)", static_cast<int>(pOwner->m_vecInputs.size()),
        static_cast<int>(pOwner->m_vecOutputs.size()));

    return true;
}

bool OvBackend::BindBuffers(InferBase* pOwner) {
    // 将任务类分配的 host 缓冲以零拷贝方式包裹为 ov::Tensor 并绑定到 infer request
    for (std::size_t i = 0; i < pOwner->m_vecInputs.size(); i++) {
        const IoMeta& c_meta = pOwner->m_vecInputs[i];
        ov::Tensor    c_tensor(ov::element::f32, ToOvShape(c_meta.dims), pOwner->GetInputHostPtr(i));
        try {
            m_inferRequest.set_input_tensor(i, c_tensor);
        }
        catch (const std::exception& e) {
            LOGE("Failed to set input tensor %s: %s", c_meta.strName.c_str(), e.what());
            return false;
        }
    }
    for (std::size_t i = 0; i < pOwner->m_vecOutputs.size(); i++) {
        const IoMeta& c_meta = pOwner->m_vecOutputs[i];
        ov::Tensor    c_tensor(ov::element::f32, ToOvShape(c_meta.dims), pOwner->GetOutputHostPtr(i));
        try {
            m_inferRequest.set_output_tensor(i, c_tensor);
        }
        catch (const std::exception& e) {
            LOGE("Failed to set output tensor %s: %s", c_meta.strName.c_str(), e.what());
            return false;
        }
    }
    return true;
}

bool OvBackend::Enqueue(InferBase* pOwner) {
    pOwner->m_cTimer.StartCpu();

    const bool b_gpu_preprocess = (pOwner->m_cParams.m_eDevice == GPU);

    // Device=GPU 时预处理结果在 device 缓冲，先 D2H 同步到 host 缓冲供 OpenVINO 使用
    if (b_gpu_preprocess) {
        for (std::size_t i = 0; i < pOwner->m_vecInputs.size(); i++) {
            std::size_t n = ElementsCount(pOwner->m_vecInputs[i].dims);
            CUDA_CHECK(cudaMemcpyAsync(pOwner->GetInputHostPtr(i), pOwner->GetInputDevicePtr(i), n * sizeof(float),
                                       cudaMemcpyKind::cudaMemcpyDeviceToHost, pOwner->m_cuStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(pOwner->m_cuStream));
    }

    // 同步推理（OpenVINO 直接读写绑定的 host 缓冲）
    try {
        m_inferRequest.infer();
    }
    catch (const std::exception& e) {
        LOGE("OpenVINO infer failed: %s", e.what());
        return false;
    }

    // 输出 host → device 拷贝：任务类通用后处理统一从 device 缓冲 D2H 读取，
    // 无论预处理路径如何，推理后都需同步输出到 device 缓冲以保持后端间行为一致
    for (std::size_t i = 0; i < pOwner->m_vecOutputs.size(); i++) {
        std::size_t n = ElementsCount(pOwner->m_vecOutputs[i].dims);
        CUDA_CHECK(cudaMemcpyAsync(pOwner->GetOutputDevicePtr(i), pOwner->GetOutputHostPtr(i), n * sizeof(float),
                                   cudaMemcpyKind::cudaMemcpyHostToDevice, pOwner->m_cuStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(pOwner->m_cuStream));

    pOwner->m_cTimer.StopCpu<Timer::ms>(std::string("ov-inference(CPU)"));
    return true;
}

void OvBackend::Release() {
    m_inferRequest  = ov::InferRequest();
    m_compiledModel = ov::CompiledModel();
    m_pModel.reset();
}

ov::Shape OvBackend::ToOvShape(const Dims& cDims) {
    ov::Shape shape;
    for (int i = 0; i < cDims.nbDims; i++) {
        shape.emplace_back(static_cast<std::size_t>(cDims.d[i]));
    }
    return shape;
}

std::size_t OvBackend::ElementsCount(const Dims& cDims) {
    std::size_t n = 1;
    for (int i = 0; i < cDims.nbDims; i++) {
        n *= static_cast<std::size_t>(cDims.d[i]);
    }
    return n;
}

#endif  // ENABLE_OPENVINO
