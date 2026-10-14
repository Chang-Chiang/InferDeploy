#include "include/tasks/Worker.h"

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <onnx_model_path> <image_path> [backend: trt|ov] [task: multi|seg]\n", argv[0]);
        return -1;
    }

    std::string str_onnx_path = argv[1];
    std::string str_img_path  = argv[2];

    // DEBUG(5) 即可看到 Timer 输出（LOGV = VERB(4)，过滤器为 eLevel <= m_eLevel）
    auto e_level  = Level::DEBUG;
    auto c_params = Params();

    c_params.m_cImgInfo = {640, 640, 3};
    c_params.m_eTask    = MULTITASK;
    c_params.m_eDevice  = CPU;
    c_params.m_ePrec    = FP16;

    // 可选第 3 参数：推理后端（默认 TensorRT；OpenVINO 需 ENABLE_OPENVINO=1 编译）
    if (argc >= 4) {
        std::string str_backend = argv[3];
        if (str_backend == "ov") {
            c_params.m_eBackend = OPENVINO;
            c_params.m_ePrec    = FP32;  // OpenVINO 后端直接用 ONNX，不走 TRT engine 构建
        }
        else {
            c_params.m_eBackend = TENSORRT;
        }
    }

    // YOLOv8-seg 分割头精度敏感层，强制使用 FP32
    c_params.m_vecFp32Layers = {"/model.22/proto/cv3/act/Mul"};

    // 可选第 4 参数：任务类型（默认 multi 检测+分割；seg 肝脏 UNet 分割）
    if (argc >= 5 && std::string(argv[4]) == "seg") {
        c_params.m_eTask   = SEGMENTATION;
        c_params.m_eDevice = CPU;
        c_params.m_ePrec   = FP32;  // 医学分割模型直接以 FP32 精度推理
    }

    auto p_worker = CreateWorker(str_onnx_path, e_level, c_params);

    p_worker->Inference(str_img_path);

    return 0;
}
