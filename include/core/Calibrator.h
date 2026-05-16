#pragma once

#include <NvInfer.h>
#include <vector>
#include <string>

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
 * 默认下推荐使用IInt8EntropyCalibrator2
*/
// class Int8EntropyCalibrator: public nvinfer1::IInt8MinMaxCalibrator {
class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
public:
    /// @brief 构造函数
    /// @param iBatchSize 校准时的 batch size
    /// @param strCalibrationSetPath 校准数据集路径，文件中每行是一个图像的绝对路径
    /// @param strCalibrationTablePath 校准表保存路径，calibrator 会将校准过程中计算的 scale 保存到这个文件中，后续构建引擎时会读取这个文件中的 scale 进行量化
    /// @param iInputSize 输入数据的大小，通常是输入通道数 * 输入高度 * 输入宽度
    /// @param iInputH 输入数据的高度
    /// @param iInputW 输入数据的宽度
    Int8EntropyCalibrator(
        const int& iBatchSize,
        const std::string& strCalibrationSetPath,
        const std::string& strCalibrationTablePath,
        const int& iInputSize,
        const int& iInputH,
        const int& iInputW);

    ~Int8EntropyCalibrator(){};

    /// @brief 获取 batch 大小，
    /// TensorRT 在构建引擎时会调用这个函数来获取校准数据的 batch 大小，
    /// calibrator 需要返回一个合适的 batch 大小，通常是根据 GPU 的内存大小和输入数据的大小确定的，
    /// 过大可能会导致 OOM，过小可能会导致校准时间过长
    /// @return 
    int getBatchSize() const noexcept override { return m_iBatchSize; };
    
    /// @brief 获取 batch 数据，TensorRT 会在构建引擎时调用这个函数获取校准数据，calibrator 需要将校准数据复制到 GPU 上，并将输入绑定指针指向这些数据
    /// @param pBindings TensorRT 传入的绑定指针数组，calibrator 需要将输入数据的 GPU 地址赋值给对应的绑定指针
    /// @param pNames 绑定名称数组，calibrator可以通过绑定名称来识别哪个绑定是输入，哪个是输出
    /// @param iNbBindings 绑定数量，calibrator可以通过这个参数来判断传入的绑定数量是否正确
    /// @return 返回 true 表示成功获取到 batch 数据，返回 false 表示没有更多的数据可供校准了
    bool getBatch(void* pBindings[], const char* pNames[], int iNbBindings) noexcept override;

    /// @brief 读取校准缓存，
    /// TensorRT 在构建引擎时会调用这个函数来读取之前保存的校准数据，
    /// 如果返回了有效的校准数据，TensorRT 就会跳过校准过程直接使用这些数据进行量化，
    /// 如果返回 nullptr 或者长度为 0，TensorRT 就会执行校准过程 并在校准完成后调用 writeCalibrationCache 来保存校准数据
    /// @param iLength 校准数据的长度，calibrator 需要将返回的校准数据的长度赋值给这个参数
    /// @return 
    const void* readCalibrationCache(std::size_t &iLength) noexcept override;

    /// @brief 写入校准缓存，
    /// TensorRT 在构建引擎时会调用这个函数来保存校准数据，calibrator 需要将校准数据保存到指定的文件中，
    /// 以便下次构建引擎时可以通过 readCalibrationCache 来读取这些数据
    /// @param pCache 校准数据的指针，calibrator 需要将这个指针指向的校准数据保存到文件中
    /// @param iLength 校准数据的长度，calibrator 需要根据这个长度来正确保存校准数据
    void writeCalibrationCache (const void* pCache, std::size_t iLength) noexcept override;

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