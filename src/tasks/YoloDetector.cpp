#include "include/tasks/YoloDetector.h"
#include "include/tasks/CocoLabels.hpp"
#include "include/utils/Process.h"
#include "include/utils/CudaCheck.h"
#include "include/utils/FileUtil.h"
#include <memory>


void YoloDetector::Setup(void const* pData, size_t iSize) {
    m_pRuntime     = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_cLogger));
    m_pEngine      = TrtUniquePtr<nvinfer1::ICudaEngine>(m_pRuntime->deserializeCudaEngine(pData, iSize));
    m_pContext     = TrtUniquePtr<nvinfer1::IExecutionContext>(m_pEngine->createExecutionContext());
    m_nvInputDims  = m_pContext->getBindingDimensions(0);
    m_nvOutputDims = m_pContext->getBindingDimensions(1);

    CUDA_CHECK(cudaStreamCreate(&m_cuStream));
    
    m_iInputSize  = m_cParams.m_cImgInfo.m_iHeight * m_cParams.m_cImgInfo.m_iWidth * m_cParams.m_cImgInfo.m_iChannel * sizeof(float);
    m_iOutputSize = m_cParams.m_iNumCls * sizeof(float);
    m_iImgArea    = m_cParams.m_cImgInfo.m_iHeight * m_cParams.m_cImgInfo.m_iWidth;

    CUDA_CHECK(cudaMallocHost((void**)&m_pInputMemory[0], m_iInputSize));
    CUDA_CHECK(cudaMallocHost((void**)&m_pOutputMemory[0], m_iOutputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pInputMemory[1], m_iInputSize));
    CUDA_CHECK(cudaMalloc((void**)&m_pOutputMemory[1], m_iOutputSize));

    m_pBindings[0] = m_pInputMemory[1];
    m_pBindings[1] = m_pOutputMemory[1];
}

void YoloDetector::ResetTask() {
    m_vecBbox.clear();
}

bool YoloDetector::PreprocessCpu() {

    m_matInputImage = cv::imread(m_strImagePath);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    m_cTimer.StartCpu();

    // InferTRT BuildEngine 时已根据模型自动更新 m_cParams.m_cImgInfo
    int   i_input_w  = m_matInputImage.cols;
    int   i_input_h  = m_matInputImage.rows;
    int   i_target_w = m_cParams.m_cImgInfo.m_iWidth;
    int   i_target_h = m_cParams.m_cImgInfo.m_iHeight;
    float f_scale    = std::min(float(i_target_w) / i_input_w, float(i_target_h) / i_input_h);
    int   i_new_w    = int(i_input_w * f_scale);
    int   i_new_h    = int(i_input_h * f_scale);

    WarpAffineInit(i_input_h, i_input_w, i_target_h, i_target_w);
    
    cv::Mat mat_tar(i_target_w, i_target_h, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mat_resized_img;
    cv::resize(m_matInputImage, mat_resized_img, cv::Size(i_new_w, i_new_h));

    int i_x, i_y;
    i_x = (i_new_w < i_target_w) ? (i_target_w - i_new_w) / 2 : 0;
    i_y = (i_new_h < i_target_h) ? (i_target_h - i_new_h) / 2 : 0;

    cv::Rect roi(i_x, i_y, i_new_w, i_new_h);

    cv::Mat mat_roi_tar = mat_tar(roi);
    mat_resized_img.copyTo(mat_roi_tar);

    int i_idx;
    int i_offset_ch0 = m_iImgArea * 0;
    int i_offset_ch1 = m_iImgArea * 1;
    int i_offset_ch2 = m_iImgArea * 2;

    int i_channel = m_nvInputDims.d[1];
    int i_height = m_nvInputDims.d[2];
    int i_width = m_nvInputDims.d[3];
    
    for (int i = 0; i < i_height; i++) {
        for (int j = 0; j < i_width; j++) {
            i_idx = i * i_width * i_channel + j * i_channel;
            m_pInputMemory[0][i_offset_ch2++] = mat_tar.data[pixel_idx + 0] / 255.0f;
            m_pInputMemory[0][i_offset_ch1++] = mat_tar.data[pixel_idx + 1] / 255.0f;
            m_pInputMemory[0][i_offset_ch0++] = mat_tar.data[pixel_idx + 2] / 255.0f;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_pInputMemory[1], m_pInputMemory[0], m_iInputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_cuStream));

    m_cTimer.StopCpu<Timer::ms>("preprocess(CPU)");
    return true;
}

bool YoloDetector::PreprocessGpu() {

    m_matInputImage = cv::imread(m_strImagePath);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }

    m_cTimer.StartGpu();

    int target_h = m_cParams.m_cImgInfo.m_iHeight;
    int target_w = m_cParams.m_cImgInfo.m_iWidth;

    ResizeGpu(m_matInputImage, m_pInputMemory[1],
              target_h, target_w, 
              InterpolationTactics::GPU_WARP_AFFINE);

    m_cTimer.StopGpu("preprocess(GPU)");
    return true;
}

bool YoloDetector::PostprocessCpu() {
    m_cTimer.StartCpu();

    int i_output_size = m_nvOutputDims.d[1] * m_nvOutputDims.d[2] * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(m_pOutputMemory[0], m_pOutputMemory[1], i_output_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    float f_conf_threshold = 0.25;  // 过滤 decode 时的 bbox
    float f_nms_threshold  = 0.45;  // 过滤 nms 时的 bbox

    int    i_box_cnt = m_nvOutputDims.d[1];
    int    i_class_cnt = m_nvOutputDims.d[2] - 4;
    float* p_tensor;

    float  f_cx, f_cy, f_w, f_h, obj, f_prob, f_conf;
    float  f_x0, f_y0, f_x1, f_y1;
    int    i_label;

    // decode, 获取 label 和 bbox 坐标
    for (int i = 0; i < i_box_cnt; i ++) {
        p_tensor = m_pOutputMemory[0] + i * m_nvOutputDims.d[2];
        i_label  = std::max_element(p_tensor + 4, p_tensor + 4 + i_class_cnt) - (p_tensor + 4);
        f_conf   = p_tensor[4 + i_label];
        if (f_conf < f_conf_threshold) {
            continue;
        }

        f_cx = p_tensor[0];
        f_cy = p_tensor[1];
        f_w  = p_tensor[2];
        f_h  = p_tensor[3];
        
        f_x0 = f_cx - f_w / 2;
        f_y0 = f_cy - f_h / 2;
        f_x1 = f_x0 + f_w;
        f_y1 = f_y0 + f_h;

        AffineTransformation(cAffineMatrix.m_arrReverse, f_x0, f_y0, &f_x0, &f_y0);
        AffineTransformation(cAffineMatrix.m_arrReverse, f_x1, f_y1, &f_x1, &f_y1);
        
        Bbox c_yolo_box(f_x0, f_y0, f_x1, f_y1, f_conf, i_label);
        m_vecBbox.emplace_back(c_yolo_box);
    }
    LOGD("the count of decoded Bbox is %d", m_vecBbox.size());

    // NMS
    std::vector<Bbox> vec_final_bbox;
    vec_final_bbox.reserve(m_vecBbox.size());
    std::sort(m_vecBbox.begin(), m_vecBbox.end(), 
              [](Bbox& box1, Bbox& box2) { return box1.m_fConfidence > box2.m_fConfidence; });

    for (int i = 0; i < m_vecBbox.size(); i ++) {
        if (m_vecBbox[i].m_bFlgRemove) {
            continue;
        }
        
        vec_final_bbox.emplace_back(m_vecBbox[i]);
        for (int j = i + 1; j < m_vecBbox.size(); j ++) {
            if (m_vecBbox[j].m_bFlgRemove) {
                continue;
            }

            if (m_vecBbox[i].m_iLabel == m_vecBbox[j].m_iLabel) {
                if (IouCalc(m_vecBbox[i], m_vecBbox[j]) > f_nms_threshold) {
                    m_vecBbox[j].m_bFlgRemove = true;
                }
            }
        }
    }
    LOGD("the count of Bbox after NMS is %d", vec_final_bbox.size());

    std::string str_tag = "detect-" + GetPrec(m_cParams.m_ePrec);
    m_strOutputPath = ChangePath(m_strImagePath, "../result", ".png", str_tag);

    // Draw bbox and label on image
    int   i_font_face  = 0;
    float f_font_scale = 0.001 * MIN(m_matInputImage.cols, m_matInputImage.rows);
    int   f_font_thick = 2;
    int   i_baseline;
    CocoLabels c_cocolabel;

    LOG("\tResult:");
    for (int i = 0; i < vec_final_bbox.size(); i ++){
        auto c_bbox = vec_final_bbox[i];
        auto str_name = c_cocolabel.CocoGetLabel(c_bbox.m_iLabel);
        auto cv_rec_color = c_cocolabel.CocoGetColor(c_bbox.m_iLabel);
        auto cv_txt_color = c_cocolabel.GetInverseColor(cv_rec_color);
        auto cv_txt = cv::format({"%s: %.2f%%"}, str_name.c_str(), c_bbox.m_fConfidence * 100);
        auto cv_txt_size = cv::getTextSize(cv_txt, i_font_face, f_font_scale, f_font_thick, &i_baseline);

        int i_txt_height = cv_txt_size.height + i_baseline + 10;
        int i_txt_width  = cv_txt_size.width + 3;

        cv::Point cv_txt_pos(round(c_bbox.m_fX0), round(c_bbox.m_fY0 - (cv_txt_size.height - i_baseline + f_font_thick)));
        cv::Rect  cv_txt_rec(round(c_bbox.m_fX0 - f_font_thick), round(c_bbox.m_fY0 - i_txt_height), i_txt_width, i_txt_height);
        cv::Rect  cv_box_rec(round(c_bbox.m_fX0), round(c_bbox.m_fY0), round(c_bbox.m_fX1 - c_bbox.m_fX0), round(c_bbox.m_fY1 - c_bbox.m_fY0));

        cv::rectangle(m_matInputImage, cv_box_rec, cv_rec_color, 3);
        cv::rectangle(m_matInputImage, cv_txt_rec, cv_rec_color, -1);
        cv::putText(m_matInputImage, cv_txt, cv_txt_pos, i_font_face, f_font_scale, cv_txt_color, f_font_thick, 16);

        LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
            str_name.c_str(), c_bbox.m_fConfidence * 100, c_bbox.m_fX0, c_bbox.m_fY0, c_bbox.m_fX1, c_bbox.m_fY1);

    }
    LOG("\tSummary:");
    LOG("\t\tDetected Objects: %d", vec_final_bbox.size());
    LOG("");

    m_cTimer.StopCpu<Timer::ms>("postprocess(CPU)");

    cv::imwrite(m_strOutputPath, m_matInputImage);
    LOG("\tsave image to %s\n", m_strOutputPath.c_str());

    return true;
}


bool YoloDetector::PostprocessGpu() {
    return PostprocessCpu();
}


std::unique_ptr<YoloDetector> MakeDetector(
    std::string strOnnxPath, Level cLevel, Params cParams)
{
    return std::make_unique<YoloDetector>(strOnnxPath, cLevel, cParams);
}
