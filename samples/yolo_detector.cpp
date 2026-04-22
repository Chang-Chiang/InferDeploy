#include "include/tasks/Worker.h"
#include "include/common/ModelParam.h"
int main(int argc, char const *argv[])
{
    std::string str_onnx_path = "models/onnx/yolov8x.onnx";

    auto e_level  = VERB;

    auto c_params = Params();
    c_params.m_cImgInfo   = {640, 640, 3};
    c_params.m_eTask      = DETECTION;
    c_params.m_eDevice    = GPU;
    c_params.m_ePrecision = FP16;

    auto p_worker = CreateWorker(str_onnx_path, e_level, c_params);

    p_worker->Inference("data/source/car.jpg");
    p_worker->Inference("data/source/crowd.jpg");
    p_worker->Inference("data/source/crossroad.jpg");
    p_worker->Inference("data/source/airport.jpg");
    p_worker->Inference("data/source/bedroom.jpg");

    return 0;
}