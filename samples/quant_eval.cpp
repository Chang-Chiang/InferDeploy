/*
 * FP32/INT8 量化精度对比评估工具
 *
 * 原理：以 FP32 引擎输出为基准，同一组图片分别经过 FP32 与 INT8 引擎推理，
 *       对比两者输出的差异，量化 INT8 的精度损失。
 *       （无需标注数据；若需绝对精度指标 mAP/Top-1 Accuracy，需接有标注的数据集，见 docs/量化.md）
 *
 * 指标：
 *   - 分类(CLASSIFICATION): top-1 一致率、平均置信度差
 *   - 检测/多任务(DETECTION/MULTITASK): 检测框匹配率（同类别 IoU>=0.5 贪心匹配）、
 *     匹配框平均 IoU、平均置信度差
 *
 * 验收线（默认值，可按项目调整）：
 *   - 分类: top-1 一致率 >= 99%，平均置信度差 <= 0.05
 *   - 检测/多任务: 匹配率 >= 95%，匹配框平均 IoU >= 0.90，平均置信度差 <= 0.05
 *
 * 用法:
 *   ./bin/trt-quant-eval <cls|det|multi> <onnx_model_path> <image_list.txt 或单张图片路径>
 *
 * 注意:
 *   - INT8 引擎构建依赖校准集（Params.m_strCalibListPath，默认 ../calibration/calibration_list.txt），
 *     需在包含 calibration/ 目录的相对路径下运行；
 *   - 校准预处理已与各任务推理预处理保持一致（分类: BILINEAR+mean/std；检测/多任务: WARP_AFFINE letterbox）；
 *   - 更换模型或校准器后需删除 ../calibration_table_*.txt 重新校准。
*/
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

#include "include/tasks/Worker.h"
#include "include/utils/FileUtil.h"

/// 验收线
static constexpr float kClsAgreeMin     = 0.99f;  // 分类 top-1 一致率下限
static constexpr float kDetMatchRateMin = 0.95f;  // 检测框匹配率下限
static constexpr float kDetMeanIouMin   = 0.90f;  // 匹配框平均 IoU 下限
static constexpr float kConfDeltaMax    = 0.05f;  // 平均置信度差上限

struct ClsResult {
    int   i_top1;
    float f_conf;
};
struct DetResult {
    std::vector<Bbox> vec_bbox;
};

/// 贪心匹配：按置信度降序遍历基准框，为每个基准框找同类别且 IoU 最大的未匹配框（IoU >= 0.5）
struct MatchStat {
    int   i_total          = 0;     // 基准（FP32）检测框总数
    int   i_matched        = 0;     // 成功匹配的框数
    float f_iou_sum        = 0.0f;  // 匹配框 IoU 累计
    float f_conf_delta_sum = 0.0f;  // 匹配框置信度差累计
};

static MatchStat MatchBboxes(const std::vector<Bbox>& vec_ref, const std::vector<Bbox>& vec_cmp) {
    MatchStat c_stat;
    c_stat.i_total = static_cast<int>(vec_ref.size());

    std::vector<bool> vec_used(vec_cmp.size(), false);
    for (const auto& c_ref : vec_ref) {
        int   i_best     = -1;
        float f_best_iou = 0.5f;  // 匹配阈值
        for (int j = 0; j < static_cast<int>(vec_cmp.size()); j++) {
            if (vec_used[j] || vec_cmp[j].m_iLabel != c_ref.m_iLabel) {
                continue;
            }
            float f_iou = IouCalc(c_ref, vec_cmp[j]);
            if (f_iou > f_best_iou) {
                f_best_iou = f_iou;
                i_best     = j;
            }
        }
        if (i_best >= 0) {
            vec_used[i_best] = true;
            c_stat.i_matched++;
            c_stat.f_iou_sum += f_best_iou;
            c_stat.f_conf_delta_sum += std::abs(c_ref.m_fConfidence - vec_cmp[i_best].m_fConfidence);
        }
    }
    return c_stat;
}

static std::vector<std::string> LoadImageList(const std::string& str_path) {
    // 传入单张图片则直接返回，否则按列表文件读取（每行一个图片路径）
    if (str_path.find(".jpg") != std::string::npos || str_path.find(".png") != std::string::npos) {
        return {str_path};
    }
    return LoadDataList(str_path);
}

int main(int argc, char const* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <cls|det|multi> <onnx_model_path> <image_list.txt 或图片路径>\n", argv[0]);
        return -1;
    }

    std::string str_task = argv[1];
    std::string str_onnx = argv[2];
    std::string str_imgs = argv[3];

    Params c_params;
    if (str_task == "cls") {
        c_params.m_eTask    = CLASSIFICATION;
        c_params.m_cImgInfo = {224, 224, 3};
    }
    else if (str_task == "det") {
        c_params.m_eTask    = DETECTION;
        c_params.m_cImgInfo = {640, 640, 3};
    }
    else if (str_task == "multi") {
        c_params.m_eTask    = MULTITASK;
        c_params.m_cImgInfo = {640, 640, 3};
        // YOLOv8-seg 分割头精度敏感层，强制使用 FP32（与 sample.cpp 保持一致）
        c_params.m_vecFp32Layers = {"/model.22/proto/cv3/act/Mul"};
    }
    else {
        printf("Unknown task type: %s (support: cls, det, multi)\n", str_task.c_str());
        return -1;
    }

    // 可选第 4 参数 "cpu"：强制 CPU 预处理路径，用于隔离 GPU 预处理核函数问题
    if (argc >= 5 && std::string(argv[4]) == "cpu") {
        c_params.m_eDevice = CPU;
        printf("Debug: force CPU preprocess\n");
    }

    // FP32 基准引擎 与 INT8 评估引擎（engine 路径按精度后缀自动区分）
    Params c_params_fp32  = c_params;
    c_params_fp32.m_ePrec = FP32;
    Params c_params_int8  = c_params;
    c_params_int8.m_ePrec = INT8;

    // INT8 优先使用显式量化模型 <base>-int8.onnx（由 data/models/quantize_yolo.py 生成），
    // 不存在时回退到同一 onnx 走 legacy calibrator 路径（仅 TRT 8.x 环境可用）
    std::string str_onnx_int8 = str_onnx;
    std::string str_quantized = str_onnx.substr(0, str_onnx.rfind(".onnx")) + "-int8.onnx";
    if (IsFileExist(str_quantized)) {
        str_onnx_int8 = str_quantized;
        printf("INT8 worker uses explicitly quantized model: %s\n", str_onnx_int8.c_str());
    }

    auto e_level = Level::INFO;
    auto p_fp32  = CreateWorker(str_onnx, e_level, c_params_fp32);
    auto p_int8  = CreateWorker(str_onnx_int8, e_level, c_params_int8);

    auto vec_images = LoadImageList(str_imgs);
    if (vec_images.empty()) {
        printf("No images found: %s\n", str_imgs.c_str());
        return -1;
    }

    // 统计量
    int       i_total         = 0;
    int       i_cls_agree     = 0;
    float     f_cls_delta_sum = 0.0f;
    MatchStat c_det_stat;

    printf("\n%-8s | %-28s | %-28s\n", "image", "FP32", "INT8");
    printf("---------+------------------------------+------------------------------\n");

    for (const auto& str_img : vec_images) {
        // FP32 基准推理
        p_fp32->Inference(str_img);
        // INT8 推理
        p_int8->Inference(str_img);

        if (c_params.m_eTask == CLASSIFICATION) {
            auto c_res_fp32 = p_fp32->m_pYoloClassifier->GetResult();
            auto c_res_int8 = p_int8->m_pYoloClassifier->GetResult();

            i_total++;
            f_cls_delta_sum += std::abs(c_res_fp32.second - c_res_int8.second);
            bool b_agree = (c_res_fp32.first == c_res_int8.first);
            if (b_agree) {
                i_cls_agree++;
            }

            printf("%-8s | top1=%-5d conf=%.4f      | top1=%-5d conf=%.4f      %s\n", GetFileName(str_img).c_str(),
                   c_res_fp32.first, c_res_fp32.second, c_res_int8.first, c_res_int8.second,
                   b_agree ? "" : "  <-- MISMATCH");
        }
        else {
            std::vector<Bbox> vec_fp32, vec_int8;
            if (c_params.m_eTask == DETECTION) {
                vec_fp32 = p_fp32->m_pYoloDetector->GetFinalBboxes();
                vec_int8 = p_int8->m_pYoloDetector->GetFinalBboxes();
            }
            else {
                vec_fp32 = p_fp32->m_pYoloMultitask->GetFinalBboxes();
                vec_int8 = p_int8->m_pYoloMultitask->GetFinalBboxes();
            }

            i_total++;
            auto c_stat = MatchBboxes(vec_fp32, vec_int8);
            c_det_stat.i_total += c_stat.i_total;
            c_det_stat.i_matched += c_stat.i_matched;
            c_det_stat.f_iou_sum += c_stat.f_iou_sum;
            c_det_stat.f_conf_delta_sum += c_stat.f_conf_delta_sum;

            printf("%-8s | boxes=%-3d                     | boxes=%-3d                     | matched=%d\n",
                   GetFileName(str_img).c_str(), static_cast<int>(vec_fp32.size()), static_cast<int>(vec_int8.size()),
                   c_stat.i_matched);
        }
    }

    // 汇总与验收判定
    printf("\n==================== QUANTIZATION EVALUATION REPORT ====================\n");

    bool b_pass = false;
    if (c_params.m_eTask == CLASSIFICATION) {
        float f_agree = i_total ? static_cast<float>(i_cls_agree) / i_total : 0.0f;
        float f_delta = i_total ? f_cls_delta_sum / i_total : 0.0f;
        b_pass        = (f_agree >= kClsAgreeMin) && (f_delta <= kConfDeltaMax);

        printf("Task:        CLASSIFICATION\n");
        printf("Images:      %d\n", i_total);
        printf("Top-1 agree: %.2f%%   (accept >= %.2f%%)\n", f_agree * 100, kClsAgreeMin * 100);
        printf("Conf delta:  %.4f  (accept <= %.4f)\n", f_delta, kConfDeltaMax);
    }
    else {
        float f_match = c_det_stat.i_total ? static_cast<float>(c_det_stat.i_matched) / c_det_stat.i_total : 0.0f;
        float f_iou   = c_det_stat.i_matched ? c_det_stat.f_iou_sum / c_det_stat.i_matched : 0.0f;
        float f_delta = c_det_stat.i_matched ? c_det_stat.f_conf_delta_sum / c_det_stat.i_matched : 0.0f;
        b_pass        = (f_match >= kDetMatchRateMin) && (f_iou >= kDetMeanIouMin) && (f_delta <= kConfDeltaMax);

        printf("Task:        %s\n", c_params.m_eTask == DETECTION ? "DETECTION" : "MULTITASK");
        printf("Images:      %d\n", i_total);
        printf("Boxes:       %d (FP32 baseline), %d matched\n", c_det_stat.i_total, c_det_stat.i_matched);
        printf("Match rate:  %.2f%%   (accept >= %.2f%%, IoU >= 0.5, same label)\n", f_match * 100,
               kDetMatchRateMin * 100);
        printf("Mean IoU:    %.4f  (accept >= %.4f, matched boxes)\n", f_iou, kDetMeanIouMin);
        printf("Conf delta:  %.4f  (accept <= %.4f, matched boxes)\n", f_delta, kConfDeltaMax);
    }

    printf("========================================================================\n");
    printf("Result:      %s\n", b_pass ? "PASS" : "FAIL");
    printf("========================================================================\n\n");

    return b_pass ? 0 : 1;
}
