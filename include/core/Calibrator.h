#pragma once

#include <NvInfer.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <fstream>
#include <iterator>

#include "include/utils/CudaCheck.h"
#include "include/utils/Logger.h"
#include "include/utils/Process.h"
#include "include/utils/FileUtil.h"
#include "include/common/ModelParam.h"

/*
 * 自定义 calibrator 类
 * 创建calibrator 时需要继承 nvinfer1 的 calibrator 类
 * TensorRT提供了五种 Calibrator 类
 *
 *   - nvinfer1::IInt8EntropyCalibrator2
 *   - nvinfer1::IInt8MinMaxCalibrator
 *   - nvinfer1::IInt8EntropyCalibrator
 *   - nvinfer1::IInt8LegacyCalibrator
 *   - nvinfer1::IInt8Calibrator
 *
 * 默认下推荐使用IInt8EntropyCalibrator2；
 * 但对 YOLOv8 这类 DWConv 较多的架构，Entropy 校准易掉点，
 * 可切换 IInt8MinMaxCalibrator 恢复精度（参考 tensorrt_starter 7.4-quantization-analysis 的结论）
 *
 * 校准注意事项（必须遵守）：
 *   1. getBatch 中的预处理必须与真实推理的预处理完全一致，否则统计到的 dynamic range 不准；
 *   2. 校准集应选取覆盖部署场景分布的真实数据集子集（如 ImageNet val / COCO val 的抽样）；
 *   3. 更换模型、校准器或校准集后，必须删除 calibration table 重新校准，否则会报错或精度异常。
*/
template <typename Base>
class Int8CalibratorBase : public Base {
public:
    /// @brief 构造函数
    /// @param cParams 模型构建参数（任务类型决定校准预处理策略，ImgInfo 决定输入尺寸）
    /// @param strCalibrationSetPath 校准数据列表路径，文件中每行是一个图像的路径
    /// @param strCalibrationTablePath 校准表保存路径，校准得到的各层 dynamic range 会写入该文件，
    ///        后续构建引擎时读取该文件跳过校准过程
    Int8CalibratorBase(const Params& cParams, const std::string& strCalibrationSetPath,
                       const std::string& strCalibrationTablePath)
        : m_iBatchSize(cParams.m_iCalibBatchSize)
        , m_cParams(cParams)
        , m_iInputH(cParams.m_cImgInfo.m_iHeight)
        , m_iInputW(cParams.m_cImgInfo.m_iWidth)
        , m_iInputSize(cParams.m_cImgInfo.m_iHeight * cParams.m_cImgInfo.m_iWidth * cParams.m_cImgInfo.m_iChannel)
        , m_iInputCount(cParams.m_iCalibBatchSize * m_iInputSize)
        , m_strCalibrationTablePath(strCalibrationTablePath) {
        m_vecImageList = LoadDataList(strCalibrationSetPath);
        // 图片数量需为 batch size 的整数倍，多余的图片裁掉
        m_vecImageList.resize(static_cast<int>(m_vecImageList.size() / m_iBatchSize) * m_iBatchSize);
        std::shuffle(m_vecImageList.begin(), m_vecImageList.end(), std::mt19937(0));
        CUDA_CHECK(cudaMalloc((void**)&m_pDeviceInput, m_iInputCount * sizeof(float)));
    }

    ~Int8CalibratorBase() {
        if (m_pDeviceInput) {
            cudaFree(m_pDeviceInput);
            m_pDeviceInput = nullptr;
        }
    };

    /// @brief 获取 batch 大小，
    /// TensorRT 在构建引擎时会调用这个函数来获取校准数据的 batch 大小，
    /// calibrator 需要返回一个合适的 batch 大小，通常是根据 GPU 的内存大小和输入数据的大小确定的，
    /// 过大可能会导致 OOM，过小可能会导致校准时间过长
    /// @return
    int getBatchSize() const noexcept override {
        return m_iBatchSize;
    };

    /// @brief 获取 batch 数据，TensorRT 会在构建引擎时调用这个函数获取校准数据，calibrator 需要将校准数据复制到 GPU 上，并将输入绑定指针指向这些数据
    /// @param pBindings TensorRT 传入的绑定指针数组，calibrator 需要将输入数据的 GPU 地址赋值给对应的绑定指针
    /// @param pNames 绑定名称数组，calibrator可以通过绑定名称来识别哪个绑定是输入，哪个绑定是输出
    /// @param iNbBindings 绑定数量，calibrator可以通过这个参数来判断传入的绑定数量是否正确
    /// @return 返回 true 表示成功获取到 batch 数据，返回 false 表示没有更多的数据可供校准了
    bool getBatch(void* pBindings[], const char* pNames[], int iNbBindings) noexcept override {
        if (m_iImageIndex + m_iBatchSize > static_cast<int>(m_vecImageList.size())) {
            return false;
        }

        LOG("%3d/%3d (%3dx%3d): %s", m_iImageIndex + 1, static_cast<int>(m_vecImageList.size()), m_iInputH, m_iInputW,
            m_vecImageList.at(m_iImageIndex).c_str());

        // 对 batch 中的每张图像进行预处理
        // 预处理策略必须与对应任务真实推理时的预处理保持一致：
        //   - 分类任务: GPU_BILINEAR + ImageNet 均值方差归一化，对应 YoloClassifier::PreprocessGpu
        //   - 检测/多任务: GPU_WARP_AFFINE letterbox + /255，对应 YoloDetector/YoloMultitask::PreprocessGpu
        // TODO 封装为函数
        cv::Mat mat_input_image;
        float   arr_mean[] = {0.406, 0.456, 0.485};
        float   arr_std[]  = {0.225, 0.224, 0.229};

        for (int i = 0; i < m_iBatchSize; i++) {
            mat_input_image = cv::imread(m_vecImageList.at(m_iImageIndex++));
            if (mat_input_image.data == nullptr) {
                LOGE("calibration image load failed: %s", m_vecImageList.at(m_iImageIndex - 1).c_str());
                return false;
            }

            if (m_cParams.m_eTask == CLASSIFICATION) {
                ResizeGpu(mat_input_image, m_pDeviceInput + i * m_iInputSize, m_iInputH, m_iInputW, arr_mean, arr_std,
                          InterpolationTactics::GPU_BILINEAR);
            }
            else {
                ResizeGpu(mat_input_image, m_pDeviceInput + i * m_iInputSize, m_iInputH, m_iInputW,
                          InterpolationTactics::GPU_WARP_AFFINE);
            }
        }

        pBindings[0] = m_pDeviceInput;

        return true;
    }

    /// @brief 读取校准缓存，
    /// TensorRT 在构建引擎时会调用这个函数来读取之前保存的校准数据，
    /// 如果返回了有效的校准数据，TensorRT 就会跳过校准过程直接使用这些数据进行量化，
    /// 如果返回 nullptr 或者长度为 0，TensorRT 就会执行校准过程 并在校准完成后调用 writeCalibrationCache 来保存校准数据
    /// @param iLength 校准数据的长度，calibrator 需要将返回的校准数据的长度赋值给这个参数
    /// @return
    const void* readCalibrationCache(std::size_t& iLength) noexcept override {
        void* p_output;
        m_vecCalibrationCache.clear();

        std::ifstream if_input(m_strCalibrationTablePath, std::ios::binary);
        if_input >> std::noskipws;
        if (m_bReadCache && if_input.good()) {
            std::copy(std::istream_iterator<char>(if_input), std::istream_iterator<char>(),
                      std::back_inserter(m_vecCalibrationCache));
        }

        iLength = m_vecCalibrationCache.size();
        if (iLength) {
            LOG("Using cached calibration table to build INT8 trt engine...");
            p_output = &m_vecCalibrationCache[0];
        }
        else {
            LOG("Creating new calibration table to build INT8 trt engine...");
            p_output = nullptr;
        }
        return p_output;
    }

    /// @brief 写入校准缓存，
    /// TensorRT 在构建引擎时会调用这个函数来保存校准数据，calibrator 需要将校准数据保存到指定的文件中，
    /// 以便下次构建引擎时可以通过 readCalibrationCache 来读取这些数据
    /// @param pCache 校准数据的指针，calibrator 需要将这个指针指向的校准数据保存到文件中
    /// @param iLength 校准数据的指针指向的数据长度
    void writeCalibrationCache(const void* pCache, std::size_t iLength) noexcept override {
        std::ofstream of_output(m_strCalibrationTablePath, std::ios::binary);
        of_output.write(reinterpret_cast<const char*>(pCache), iLength);
        of_output.close();
    }

private:
    const int         m_iBatchSize;
    const int         m_iInputH;
    const int         m_iInputW;
    const int         m_iInputSize;
    const int         m_iInputCount;
    const Params      m_cParams;
    const std::string m_strCalibrationTablePath;

    std::vector<std::string> m_vecImageList;
    std::vector<char>        m_vecCalibrationCache;

    float* m_pDeviceInput{nullptr};
    bool   m_bReadCache{true};
    int    m_iImageIndex{0};
};

/// Entropy 校准器：基于 KL 熵最小化，通用性强（默认推荐）
using Int8EntropyCalibrator = Int8CalibratorBase<nvinfer1::IInt8EntropyCalibrator2>;
/// MinMax 校准器：保留 FP32 极值分布，YOLOv8 这类含 DWConv 架构上精度恢复更好
using Int8MinMaxCalibrator = Int8CalibratorBase<nvinfer1::IInt8MinMaxCalibrator>;
