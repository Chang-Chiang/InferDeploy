#include "include/tasks/YoloMultitask.h"
#include "include/tasks/CocoLabels.hpp"
#include "include/utils/Process.h"
#include "include/utils/CudaCheck.h"
#include "include/utils/FileUtil.h"


void YoloMultitask::Setup(void const* pData, size_t iSize) {
    m_pRuntime      = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_pLogger), DestroyTrtPtr<nvinfer1::IRuntime>);
    m_pEngine       = std::shared_ptr<nvinfer1::ICudaEngine>(m_pRuntime->deserializeCudaEngine(pData, iSize), DestroyTrtPtr<nvinfer1::ICudaEngine>);
    m_pContext      = std::shared_ptr<nvinfer1::IExecutionContext>(m_pEngine->createExecutionContext(), DestroyTrtPtr<nvinfer1::IExecutionContext>);
    m_nvInputDims   = m_pContext->getBindingDimensions(0);
    m_nvSegmentDims = m_pContext->getBindingDimensions(1);
    m_nvDetectDims  = m_pContext->getBindingDimensions(2);

    CUDA_CHECK(cudaStreamCreate(&m_cuStream));
    
    m_iInputSize   = m_pParams->m_cImgInfo.m_iHeight * m_pParams->m_cImgInfo.m_iWidth * m_pParams->m_cImgInfo.m_iChannel * sizeof(float);
    m_iImgArea     = m_pParams->m_cImgInfo.m_iHeight * m_pParams->m_cImgInfo.m_iWidth;
    m_iDetectSize  = m_nvDetectDims.d[1] * m_nvDetectDims.d[2] * sizeof(float);
    m_iSegmentSize = m_nvSegmentDims.d[1] * m_nvSegmentDims.d[2] * m_nvSegmentDims.d[3] * sizeof(float);

    CUDA_CHECK(cudaMallocHost(&m_pInputMemory[0], m_iInputSize));
    CUDA_CHECK(cudaMallocHost(&m_pDetectMemory[0], m_iDetectSize));
    CUDA_CHECK(cudaMallocHost(&m_pSegmentMemory[0], m_iSegmentSize));
    CUDA_CHECK(cudaMalloc(&m_pInputMemory[1], m_iInputSize));
    CUDA_CHECK(cudaMalloc(&m_pDetectMemory[1], m_iDetectSize));
    CUDA_CHECK(cudaMalloc(&m_pSegmentMemory[1], m_iSegmentSize));

    m_pBindings[0] = m_pInputMemory[1];
    m_pBindings[1] = m_pSegmentMemory[1];
    m_pBindings[2] = m_pDetectMemory[1];
}

void YoloMultitask::ResetTask(){
    for (int i = 0; i < m_vecBbox.size(); i++) {
        m_vecBbox[i].m_matBoxMask.release();
        m_vecBbox[i].m_matMc.release();
    }
    m_vecBbox.clear();
    m_matMask.release();
}

bool YoloMultitask::PreprocessCpu() {

    m_matInputImage = cv::imread(m_strImagePath);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    m_pTimer->StartCpu();

    int   i_input_w  = m_matInputImage.cols;
    int   i_input_h  = m_matInputImage.rows;
    int   i_target_w = m_pParams->m_cImgInfo.m_iWidth;
    int   i_target_h = m_pParams->m_cImgInfo.m_iHeight;
    float f_scale    = std::min(float(i_target_w)/i_input_w, float(i_target_h)/i_input_h);
    int   i_new_w    = int(i_input_w * f_scale);
    int   i_new_h    = int(i_input_h * f_scale);

    WarpAffineInit(i_input_h, i_input_w, i_target_h, i_target_w);
    
    cv::Mat mat_tar(i_target_w, i_target_h, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mat_resized_img;
    cv::resize(m_matInputImage, mat_resized_img, cv::Size(i_new_w, i_new_h));

    int x, y;
    x = (i_new_w < i_target_w) ? (i_target_w - i_new_w) / 2 : 0;
    y = (i_new_h < i_target_h) ? (i_target_h - i_new_h) / 2 : 0;

    cv::Rect cv_roi(x, y, i_new_w, i_new_h);

    cv::Mat mat_roi_tar = mat_tar(cv_roi);
    mat_resized_img.copyTo(mat_roi_tar);

    // 添加边界检查以避免数组越界
    int i_height = m_nvInputDims.d[2];
    int i_width = m_nvInputDims.d[3];
    int i_channel = m_nvInputDims.d[1];
    if(i_height != i_target_h || i_width != i_target_w) {
        LOGW("Warning: Input dimensions mismatch. Expected: %dx%d, Got: %dx%d", 
             i_target_w, i_target_h, i_width, i_height);

        i_height = m_nvInputDims.d[2];
        i_width = m_nvInputDims.d[3];
    }

    int i_idx;
    int i_offset_ch0 = m_iImgArea * 0;
    int i_offset_ch1 = m_iImgArea * 1;
    int i_offset_ch2 = m_iImgArea * 2;
    for (int i = 0; i < m_nvInputDims.d[2]; i++) {
        for (int j = 0; j < m_nvInputDims.d[3]; j++) {
            i_idx = i * m_nvInputDims.d[3] * m_nvInputDims.d[1] + j * m_nvInputDims.d[1];
            m_pInputMemory[0][i_offset_ch2++] = mat_roi_tar.data[i_idx + 0] / 255.0f;
            m_pInputMemory[0][i_offset_ch1++] = mat_roi_tar.data[i_idx + 1] / 255.0f;
            m_pInputMemory[0][i_offset_ch0++] = mat_roi_tar.data[i_idx + 2] / 255.0f;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(m_pInputMemory[1], m_pInputMemory[0], m_iInputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_cuStream));

    m_pTimer->StopCpu<Timer::ms>("preprocess(CPU)");
    return true;
}

bool YoloMultitask::PreprocessGpu() {
    m_matInputImage = cv::imread(m_strImagePath);
    if (m_matInputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }

    m_pTimer->StartGpu();

    ResizeGpu(m_matInputImage, m_pInputMemory[1],
              m_pParams->m_cImgInfo.m_iHeight, m_pParams->m_cImgInfo.m_iWidth, 
              InterpolationTactics::GPU_WARP_AFFINE);

    m_pTimer->StopGpu("preprocess(GPU)");
    return true;
}


bool YoloMultitask::PostprocessCpu() {
    m_pTimer->StartCpu();

    CUDA_CHECK(cudaMemcpyAsync(m_pDetectMemory[0], m_pDetectMemory[1], m_iDetectSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaMemcpyAsync(m_pSegmentMemory[0], m_pSegmentMemory[1], m_iSegmentSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    float f_conf_threshold = 0.25;
    float f_nms_threshold  = 0.45;

    int    i_mc_cnt    = 32;
    int    i_box_cnt = m_nvDetectDims.d[1];
    int    i_class_cnt = m_nvDetectDims.d[2] - 4 - i_mc_cnt;
    float* p_feat;
    float* p_mc;

    float  f_cx, f_cy, f_w, f_h, f_obj, f_prob, f_conf;
    float  f_x0, f_y0, f_x1, f_y1;
    int    i_label;
    std::vector<Bbox> vec_bbox;

    for (int i = 0; i < i_box_cnt; i ++) {
        p_feat  = m_pDetectMemory[0] + i * m_nvDetectDims.d[2];
        p_mc    = p_feat + 4 + i_class_cnt;
        i_label = std::max_element(p_feat + 4, p_feat + 4 + i_class_cnt) - (p_feat + 4);

        f_conf  = p_feat[4 + i_label];
        if (f_conf < f_conf_threshold) {
            continue;
        }

        f_cx = p_feat[0];
        f_cy = p_feat[1];
        f_w  = p_feat[2];
        f_h  = p_feat[3];
        
        f_x0 = f_cx - f_w / 2;
        f_y0 = f_cy - f_h / 2;
        f_x1 = f_x0 + f_w;
        f_y1 = f_y0 + f_h;

        AffineTransformation(cAffineMatrix.m_arrReverse, f_x0, f_y0, &f_x0, &f_y0);
        AffineTransformation(cAffineMatrix.m_arrReverse, f_x1, f_y1, &f_x1, &f_y1);
        
        Bbox c_yolo_box(f_x0, f_y0, f_x1, f_y1, f_conf, i_label, m_matInputImage);
        c_yolo_box.m_matMc = cv::Mat(1, 32, CV_32F, p_mc);
        vec_bbox.emplace_back(c_yolo_box);
    }
    LOGD("the count of decoded Bbox is %d", vec_bbox.size());

    std::sort(vec_bbox.begin(), vec_bbox.end(), 
              [](Bbox& box1, Bbox& box2){return box1.m_fConfidence > box2.m_fConfidence;});

    cv::Mat mat_mask_in;
    m_vecBbox.reserve(vec_bbox.size());
    for(int i = 0; i < vec_bbox.size(); i ++) {
        if (vec_bbox[i].m_bFlgRemove) {
            continue;
        }
        
        m_vecBbox.emplace_back(vec_bbox[i]);
        mat_mask_in.push_back(vec_bbox[i].m_matMc);

        for (int j = i + 1; j < vec_bbox.size(); j ++) {
            if (vec_bbox[j].m_bFlgRemove) {
                continue;
            }

            if (vec_bbox[i].m_iLabel == vec_bbox[j].m_iLabel) {
                if (IouCalc(vec_bbox[i], vec_bbox[j]) > f_nms_threshold) {
                    vec_bbox[j].m_bFlgRemove = true;
                }
            }
        }
    }
    LOGD("the count of Bbox after NMS is %d", m_vecBbox.size());

    cv::Mat mat_proto = cv::Mat(m_nvSegmentDims.d[1], m_nvSegmentDims.d[2] * m_nvSegmentDims.d[3], CV_32F, m_pSegmentMemory[0]);
    cv::Mat mat_mul_res = (mat_mask_in * mat_proto).t();  //得到一个[25600, n]的矩阵
    cv::Mat mat_mask = mat_mul_res.reshape(m_vecBbox.size(), {m_nvSegmentDims.d[2], m_nvSegmentDims.d[3]});

    std::vector<cv::Mat> vec_mask_channel;
    cv::split(mat_mask, vec_mask_channel);  //分割成每一个box的mask来分别处理, maskChannels的大小为(1 x 160 x 160)

    cv::Rect cv_roi;
    if (m_matInputImage.rows > m_matInputImage.cols) {
        cv_roi = cv::Rect(0, 0, 160 * m_matInputImage.cols / m_matInputImage.rows, 160);
        cv_roi.x = (160 - cv_roi.width) / 2;
    }
    else {
        cv_roi = cv::Rect(0, 0, 160, 160 * m_matInputImage.rows / m_matInputImage.cols);
        cv_roi.y = (160 - cv_roi.height) / 2;
    }

    for (int i = 0; i < m_vecBbox.size(); i ++) {
        cv::Mat mat_dest, mat_mask;

        // cpu实现的sigmoid x = 1 / (1 + exp(-x))
        cv::exp(-vec_mask_channel[i], mat_dest);
        mat_dest = 1.0 / (1.0 + mat_dest);
        mat_dest = mat_dest(cv_roi);
        cv::resize(
            mat_dest, 
            mat_mask, 
            cv::Size(m_matInputImage.cols, m_matInputImage.rows), 
            cv::INTER_LINEAR
        );
        m_vecBbox[i].m_matBoxMask = mat_mask(m_vecBbox[i].m_cvRect) > 0.5;
    }

    std::string str_tag   = "segment-" + GetPrec(m_pParams->m_ePrec);
    m_strOutputPath = ChangePath(m_strImagePath, "../result", ".png", str_tag);

    int   i_front_face  = 0;
    float f_front_scale = 0.001 * MIN(m_matInputImage.cols, m_matInputImage.rows);
    int   i_font_thick = 2;
    int   i_baseline;
    CocoLabels c_label;

    m_matMask = m_matInputImage.clone();
    for (int i = 0; i < m_vecBbox.size(); i ++) {
        auto mask_color = c_label.CocoGetColor(m_vecBbox[i].m_iLabel);
        m_matMask(m_vecBbox[i].m_cvRect).setTo(mask_color, m_vecBbox[i].m_matBoxMask);
    }
    cv::addWeighted(m_matInputImage, 0.5, m_matMask, 0.8, 1, m_matInputImage);

    LOG("\tResult:");

    for (int i = 0; i < m_vecBbox.size(); i ++){
        auto box        = m_vecBbox[i];
        auto name       = c_label.CocoGetLabel(box.m_iLabel);
        auto rec_color  = c_label.CocoGetColor(box.m_iLabel);
        auto txt_color  = c_label.GetInverseColor(rec_color);
        auto txt        = cv::format({"%s: %.2f%%"}, name.c_str(), box.m_fConfidence * 100);
        auto txt_size   = cv::getTextSize(txt, i_front_face, f_front_scale, i_font_thick, &i_baseline);
        auto txt_height = txt_size.height + i_baseline + 10;
        auto txt_width  = txt_size.width + 3;

        cv::Point txt_pos(round(box.m_fX0), round(box.m_fY0 - (txt_size.height - i_baseline + i_font_thick)));
        cv::Rect  txt_rec(round(box.m_fX0 - i_font_thick), round(box.m_fY0 - txt_height), txt_width, txt_height);
        cv::Rect  box_rec(round(box.m_fX0), round(box.m_fY0), round(box.m_fX1 - box.m_fX0), round(box.m_fY1 - box.m_fY0));

        cv::rectangle(m_matInputImage, box_rec, rec_color, 3);
        cv::rectangle(m_matInputImage, txt_rec, rec_color, -1);
        cv::putText(m_matInputImage, txt, txt_pos, i_front_face, f_front_scale, txt_color, i_font_thick, 16);

        LOG("%+20s detected. Confidence: %.2f%%. Cord: (f_x0, f_y0):(%6.2f, %6.2f), (f_x1, f_y1)(%6.2f, %6.2f)", 
            name.c_str(), box.m_fConfidence * 100, box.m_fX0, box.m_fY0, box.m_fX1, box.m_fY1);
    }
    LOG("\tSummary:");
    LOG("\t\tDetected objects: %d", m_vecBbox.size());
    LOG("");

    m_pTimer->StopCpu<Timer::ms>("postprocess(CPU)");

    cv::imwrite(m_strOutputPath, m_matInputImage);
    LOG("\tsave image to %s", m_strOutputPath.c_str());

    m_pTimer->Show();
    printf("\n");

    return true;
}


bool YoloMultitask::PostprocessGpu() {
    return PostprocessCpu();
}

std::shared_ptr<YoloMultitask> MakeMultitask(
    std::string strOnnxPath, Level cLevel, Params cParams)
{
    return std::make_shared<YoloMultitask>(strOnnxPath, cLevel, cParams);
}