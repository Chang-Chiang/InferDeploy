#include "include/core/Calibrator.h"
#include "include/utils/Process.h"
#include "include/utils/FileUtil.h"
#include "include/utils/Logger.h"
#include "include/utils/CudaCheck.h"

#include <iterator>
#include <fstream>

Int8EntropyCalibrator::Int8EntropyCalibrator(
    const int& iBatchSize,
    const std::string& strCalibrationSetPath,
    const std::string& strCalibrationTablePath,
    const int& iInputSize,
    const int& iInputH,
    const int& iInputW)
    : m_iBatchSize(iBatchSize)
    , m_iInputH(iInputH)
    , m_iInputW(iInputW)
    , m_iInputSize(iInputSize)
    , m_iInputCount(iBatchSize * iInputSize)
    , m_strCalibrationTablePath(strCalibrationTablePath)
{
    m_vecImageList = LoadDataList(strCalibrationSetPath);
    m_vecImageList.resize(static_cast<int>(m_vecImageList.size() / m_iBatchSize) * m_iBatchSize);
    std::random_shuffle(m_vecImageList.begin(), m_vecImageList.end(), 
                        [](int i){ return rand() % i; });
    CUDA_CHECK(cudaMalloc((void**)&m_pDeviceInput, m_iInputCount * sizeof(float)));
}

bool Int8EntropyCalibrator::getBatch(
    void* pBindings[], const char* pNames[], int iNbBindings) noexcept
{
    if (m_iImageIndex + m_iBatchSize >= m_vecImageList.size() + 1) {
        return false;
    }

    LOG("%3d/%3d (%3dx%3d): %s", 
        m_iImageIndex + 1, m_vecImageList.size(), m_iInputH, m_iInputW, m_vecImageList.at(m_iImageIndex).c_str());
    
    // 对 batch 中的每张图像进行预处理
    // 不同 task 预处理不同
    // TODO 封装为函数
    cv::Mat mat_input_image;
    for (int i = 0; i < m_iBatchSize; i++) {
        mat_input_image = cv::imread(m_vecImageList.at(m_iImageIndex++));
        ResizeGpu(
            mat_input_image, 
            m_pDeviceInput + i * m_iInputSize,
            m_iInputH, m_iInputW, 
            InterpolationTactics::GPU_BILINEAR_CENTER);
    }

    pBindings[0] = m_pDeviceInput;

    return true;
}
    
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& iLength) noexcept
{
    void* p_output;
    m_vecCalibrationCache.clear();

    std::ifstream if_input(m_strCalibrationTablePath, std::ios::binary);
    if_input >> std::noskipws;
    if (m_bReadCache && if_input.good())
        std::copy(std::istream_iterator<char>(if_input), std::istream_iterator<char>(), std::back_inserter(m_vecCalibrationCache));

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

void Int8EntropyCalibrator::writeCalibrationCache(const void* pCache, size_t iLength) noexcept
{
    std::ofstream of_output(m_strCalibrationTablePath, std::ios::binary);
    of_output.write(reinterpret_cast<const char*>(pCache), iLength);
    of_output.close();
}