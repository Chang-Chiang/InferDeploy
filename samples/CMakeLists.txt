cmake_minimum_required(VERSION 3.18)
project(yolo_detector)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA and TensorRT paths
set(CUDA_VER 12.2)
set(CUDA_DIR /usr/local/cuda-${CUDA_VER})
set(TENSORRT_INSTALL_DIR /home/cc/Downloads/TensorRT-8.6.1.6)
set(OPENCV_INSTALL_DIR /usr/local)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find TensorRT
find_library(TENSORRT_LIBRARY nvinfer PATHS ${TENSORRT_INSTALL_DIR}/lib)
find_library(TENSORRT_PARSER_LIBRARY nvonnxparser PATHS ${TENSORRT_INSTALL_DIR}/lib)
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDA_DIR}/lib64)
find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_DIR}/lib64)

# Include directories
# Note: Source files use "include/xxx" so we add CMAKE_SOURCE_DIR to include path
include_directories(
    ${CUDA_DIR}/include
    ${TENSORRT_INSTALL_DIR}/include
    ${OPENCV_INSTALL_DIR}/include/opencv4
    ${CMAKE_SOURCE_DIR}
)

# Link directories
link_directories(
    ${CUDA_DIR}/lib64
    ${TENSORRT_INSTALL_DIR}/lib
    ${OpenCV_LIBRARY_DIRS}
)

# Sample executable
add_executable(yolo_detector
    yolo_detector.cpp
)

# Link with infer_trt library
target_link_libraries(yolo_detector
    infer_trt
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSER_LIBRARY}
    ${OpenCV_LIBS}
    stdc++fs
)

# Set output directory
set_target_properties(yolo_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)