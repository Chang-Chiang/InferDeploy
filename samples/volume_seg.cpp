/*
 * 完整体数据逐切面分割工具
 *
 * 遍历输入目录中的切片序列（png，按文件名排序），逐切面调用框架分割任务推理，
 * 输出（逐层 mask png 不再落盘，统一写 3D 产物，见 UnetSegmenter::SetSaveDebug）：
 *   - <output_dir>/mask_volume.raw     3D 二值 mask（uint8 0/255，W x H x N，层优先）
 *   - <output_dir>/summary.txt         各切面肝区面积统计（末行为体素尺寸）
 *   - <output_dir>/post3d_in/          3D 后处理（post3d 流水线）的输入：
 *                                      slices/slice_%04d.u8 序列 + meta.txt
 *
 * 可选 --post3d：分割完成后自动调用 3D 后处理流水线（bin/liver3d）：
 *   CCL 连通域过滤 -> 3D 形态学开闭 -> 3D 高斯平滑，输出规整器官 mask
 *   - <output_dir>/post3d_out/liver_final.u8   0/1 体素 mask + 体积/Dice 统计
 *   - <output_dir>/post3d_out/liver_preview_z*.pgm
 *   - <output_dir>/post3d_out/final_slices/final_slice_%04d.png
 *     后处理结果回导的逐层 2D 切面（0/255，与输入切片序号一一对应）
 *
 * 用法:
 *   ./bin/trt-volume-seg <onnx_path> <input_slice_dir> <output_dir> [trt|ov] \
 *                    [--spacing dx dy dz] [--post3d [liver3d 额外参数]]
 *
 * 切片预处理参考: data/models/prepare_volume.py（nii -> 直方图匹配 -> png 序列）
*/
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "include/tasks/Worker.h"

int main(int argc, char const* argv[])
{
    if (argc < 4) {
        printf("Usage: %s <onnx_path> <input_slice_dir> <output_dir> [trt|ov] "
               "[--spacing dx dy dz] [--post3d [extra liver3d args...]]\n", argv[0]);
        return -1;
    }

    std::string str_onnx    = argv[1];
    std::string str_in      = argv[2];
    std::string str_out     = argv[3];
    std::string str_backend = "trt";
    float arr_spacing[3]    = {1.0f, 1.0f, 1.0f};
    bool  b_post3d          = false;
    std::string str_post3d_extra;

    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a == "trt" || a == "ov") {
            str_backend = a;
        }
        else if (a == "--spacing" && i + 3 < argc) {
            arr_spacing[0] = atof(argv[++i]);
            arr_spacing[1] = atof(argv[++i]);
            arr_spacing[2] = atof(argv[++i]);
        }
        else if (a == "--post3d") {
            b_post3d = true;
            // 收集透传给 liver3d 的额外参数（如 --keep-largest --mrx 2）
            while (i + 1 < argc && argv[i + 1][0] == '-') {
                str_post3d_extra += " ";
                str_post3d_extra += argv[++i];
            }
        }
    }

    // 收集切片（按文件名排序 = 层序）
    std::vector<std::string> vec_slices;
    for (const auto& entry : std::filesystem::directory_iterator(str_in)) {
        if (entry.path().extension() == ".png") {
            vec_slices.emplace_back(entry.path().string());
        }
    }
    std::sort(vec_slices.begin(), vec_slices.end());
    if (vec_slices.empty()) {
        printf("No slices found in %s\n", str_in.c_str());
        return -1;
    }
    printf("Volume slices: %zu\n", vec_slices.size());

    std::filesystem::create_directories(str_out);

    Params c_params;
    c_params.m_eTask    = SEGMENTATION;
    c_params.m_eDevice  = CPU;
    c_params.m_ePrec    = FP32;
    c_params.m_eBackend = (str_backend == "ov") ? OPENVINO : TENSORRT;

    auto p_worker = CreateWorker(str_onnx, Level::INFO, c_params);
    auto* p_seg   = p_worker->m_pUnetSegmenter.get();
    p_seg->SetSaveDebug(false);  // 关闭逐层落盘，统一写 3D 输出

    // 3D 后处理输入目录
    std::string str_p3d_in = str_out + "/post3d_in";
    std::filesystem::create_directories(str_p3d_in + "/slices");

    std::ofstream f_summary(str_out + "/summary.txt");
    f_summary << "slice\tliver_pixels\tliver_ratio(%)\n";

    int i_w = 0, i_h = 0;
    std::vector<uint8_t> vec_volume;
    int i_layer = 0;

    for (const auto& str_slice : vec_slices) {
        p_worker->Inference(str_slice);

        const cv::Mat& c_mask = p_seg->GetMask();
        if (c_mask.empty()) {
            printf("slice %d failed: %s\n", i_layer, str_slice.c_str());
            f_summary << i_layer << "\t-1\t-1\n";
            i_layer++;
            continue;
        }
        if (i_w == 0) {
            i_w = c_mask.cols;
            i_h = c_mask.rows;
            vec_volume.reserve(static_cast<size_t>(i_w) * i_h * vec_slices.size());
        }

        int i_pixels = cv::countNonZero(c_mask);
        vec_volume.insert(vec_volume.end(), c_mask.data, c_mask.data + static_cast<size_t>(i_w) * i_h);
        f_summary << i_layer << "\t" << i_pixels << "\t"
                  << (100.0 * i_pixels / (i_w * i_h)) << "\n";

        // 3D-Process 兼容输入: 0/255 单字节切片 + meta
        std::string str_u8 = str_p3d_in + "/slices/slice_" + std::to_string(10000 + i_layer).substr(1) + ".u8";
        std::ofstream f_slice(str_u8, std::ios::binary);
        f_slice.write(reinterpret_cast<const char*>(c_mask.data),
                      static_cast<std::streamsize>(i_w) * i_h);
        f_slice.close();
        i_layer++;
    }

    std::ofstream f_raw(str_out + "/mask_volume.raw", std::ios::binary);
    f_raw.write(reinterpret_cast<const char*>(vec_volume.data()),
                static_cast<std::streamsize>(vec_volume.size()));
    f_raw.close();

    f_summary << "volume: " << i_w << " x " << i_h << " x " << i_layer << "\n";
    f_summary.close();

    // meta.txt: W H D dx dy dz（3D-Process 输入）
    std::ofstream f_meta(str_p3d_in + "/meta.txt");
    f_meta << i_w << " " << i_h << " " << i_layer << " "
           << arr_spacing[0] << " " << arr_spacing[1] << " " << arr_spacing[2] << "\n";
    f_meta.close();

    printf("Done. %d slices, volume %d x %d x %d\n", i_layer, i_w, i_h, i_layer);
    printf("Output: %s/mask_volume.raw\n", str_out.c_str());

    // 一键 3D 后处理：CCL 连通域过滤 -> 形态学开闭 -> 高斯平滑
    if (b_post3d && i_layer > 0) {
        // 依次探测 liver3d 二进制位置（不同工作目录下）
        std::string str_liver3d;
        for (const auto& str_cand : {"./liver3d", "bin/liver3d", "../bin/liver3d", "post3d/bin/liver3d"}) {
            if (std::filesystem::exists(str_cand)) { str_liver3d = str_cand; break; }
        }
        if (str_liver3d.empty()) {
            printf("liver3d not found. Build it with: make liver3d\n");
            return -1;
        }
        // liver3d 不会自建输出目录
        std::filesystem::create_directories(str_out + "/post3d_out");
        std::string str_cmd = str_liver3d
            + " --slices " + str_p3d_in + "/slices"
            + " --meta " + str_p3d_in + "/meta.txt"
            + " --no-gt"
            + " --out " + str_out + "/post3d_out/liver"
            + str_post3d_extra;
        printf("Running 3D post-processing: %s\n", str_cmd.c_str());
        int ret = std::system(str_cmd.c_str());
        if (ret != 0) {
            printf("3D post-processing failed (exit %d). Build it with: make liver3d\n", ret);
            return ret;
        }

        // 后处理结果导出为 2D 切面序列（final_slice_%04d.png，0/255 mask）
        std::string str_final = str_out + "/post3d_out/liver_final.u8";
        std::string str_slice_dir = str_out + "/post3d_out/final_slices";
        std::filesystem::create_directories(str_slice_dir);
        std::ifstream f_in(str_final, std::ios::binary);
        if (!f_in.good()) {
            printf("warning: %s not found, skip final slice export\n", str_final.c_str());
        }
        else {
            for (int z = 0; z < i_layer; z++) {
                std::vector<uint8_t> vec_slice(static_cast<size_t>(i_w) * i_h);
                f_in.read(reinterpret_cast<char*>(vec_slice.data()),
                          static_cast<std::streamsize>(vec_slice.size()));
                cv::Mat c_slice(i_h, i_w, CV_8UC1, vec_slice.data());
                c_slice *= 255;  // 0/1 -> 0/255
                char ch_name[64];
                snprintf(ch_name, sizeof(ch_name), "final_slice_%04d.png", z);
                cv::imwrite(str_slice_dir + "/" + ch_name, c_slice);
            }
            printf("Final mask exported as 2D slices: %s/final_slice_*.png (%d slices)\n",
                   str_slice_dir.c_str(), i_layer);
        }
        printf("3D post-processing output: %s/post3d_out/liver_final.u8\n", str_out.c_str());
    }

    return 0;
}
