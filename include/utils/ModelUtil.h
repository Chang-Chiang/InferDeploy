#pragma once

#include <string>
#include <NvInfer.h>

std::string PrintTensorShape(nvinfer1::ITensor* nvTensor);

std::string GetPrecision(nvinfer1::DataType nvDataType);

std::string PrintDims(const nvinfer1::Dims nvDims);

std::string PrintTensor(float* pTensor, int iSize);
