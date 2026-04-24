#pragma once

#include <NvInfer.h>
#include <vector>
#include <string>

class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(
        const int& iBatchSize,
        const std::string& strCalibrationSetPath,
        const std::string& strCalibrationTablePath,
        const int& iInputSize,
        const int& iInputH,
        const int& iInputW);

    ~Int8EntropyCalibrator(){};

    int         getBatchSize() const noexcept override { return m_iBatchSize; };
    bool        getBatch(void* pBindings[], const char* names[], int iNbBindings) noexcept override;
    const void* readCalibrationCache(std::size_t &iLength) noexcept override;
    void        writeCalibrationCache (const void* pCache, std::size_t iLength) noexcept override;

private:
    const int   m_iBatchSize;
    const int   m_iInputH;
    const int   m_iInputW;
    const int   m_iInputSize;
    const int   m_iInputCount;
    const std::string m_strCalibrationTablePath {nullptr};
    
    std::vector<std::string> m_vecImageList;
    std::vector<char>        m_vecCalibrationCache;

    float* m_pDeviceInput{nullptr};
    bool   m_bReadCache{true};
    int    m_iImageIndex;
};