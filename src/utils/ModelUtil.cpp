#include "include/utils/ModelUtil.h"

std::string PrintTensorShape(nvinfer1::ITensor* nvTensor) {
    std::string str_res;
    str_res += "[";
    auto dims = nvTensor->getDimensions();
    for (int i = 0; i < dims.nbDims; i++) {
        str_res += std::to_string(dims.d[i]);
        if (i != dims.nbDims - 1) {
            str_res += " x ";
        }
    }
    str_res += "]";
    return str_res;
}

std::string GetPrecision(nvinfer1::DataType nvDataType) {
    switch(nvDataType) {
        case nvinfer1::DataType::kFLOAT:  return "FP32";
        case nvinfer1::DataType::kHALF:   return "FP16";
        case nvinfer1::DataType::kINT32:  return "INT32";
        case nvinfer1::DataType::kINT8:   return "INT8";
        case nvinfer1::DataType::kUINT8:  return "UINT8";
        default:                          return "unknown";
    }
}

std::string PrintDims(const nvinfer1::Dims nvDims) {
    int i_n = 0;
    char c_buff[100];
    std::string str_res;

    i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, "[ ");
    for (int i = 0; i < nvDims.nbDims; i++){
        i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, "%d", nvDims.d[i]);
        if (i != nvDims.nbDims - 1) {
            i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, ", ");
        }
    }
    i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, " ]");
    str_res = c_buff;
    return str_res;
}

std::string PrintTensor(float* pTensor, int iSize) {
    int i_n = 0;
    char c_buff[100];
    std::string str_res;
    i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, "[ ");
    for (int i = 0; i < iSize; i++){
        i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, "%8.4lf", pTensor[i]);
        if (i != iSize - 1){
            i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, ", ");
        }
    }
    i_n += snprintf(c_buff + i_n, sizeof(c_buff) - i_n, " ]");
    str_res = c_buff;
    return str_res;
}
