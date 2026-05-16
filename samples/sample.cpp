#include "include/tasks/Worker.h"

int main(int argc, char const *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <onnx_model_path> <image_path>...\n", argv[0]);
        return -1;
    }

    // std::string str_onnx_path = "data/models/onnx/yolov8n-seg.onnx";
    // std::string str_img_path  = "data/source/crowd.jpg";
    std::string str_onnx_path = argv[1];
    std::string str_img_path  = argv[2];

    auto e_level         = Level::DEBUG;
    auto c_params        = Params();

    c_params.m_cImgInfo  = {640, 640, 3};
    c_params.m_eTask     = MULTITASK;
    c_params.m_eDevice   = CPU;
    c_params.m_ePrec     = FP16;

    auto p_worker   = CreateWorker(str_onnx_path, e_level, c_params);

    p_worker->Inference(str_img_path);

    return 0;
}