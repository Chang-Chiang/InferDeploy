cmake_minimum_required(VERSION 3.18)
project(infer_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 61) 

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

if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT nvinfer not found")
endif()

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

# Source files
set(CORE_SOURCES
    src/core/InferTRT.cpp
    src/core/Calibrator.cpp
)

set(TASK_SOURCES
    src/tasks/YoloClassifier.cpp
    src/tasks/YoloDetector.cpp
    src/tasks/Worker.cpp
)

set(UTILS_SOURCES
    src/utils/Logger.cpp
    src/utils/Timer.cpp
    src/utils/FileUtil.cpp
    src/utils/ModelUtil.cpp
    src/utils/Process.cpp
    src/utils/Process.cu
)

# Create static library
add_library(infer_trt STATIC
    ${CORE_SOURCES}
    ${TASK_SOURCES}
    ${UTILS_SOURCES}
)

# Target link libraries
target_link_libraries(infer_trt
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSER_LIBRARY}
    ${OpenCV_LIBS}
    stdc++fs
)

# Set output directories
set_target_properties(infer_trt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib
)

# Add samples subdirectory
add_subdirectory(samples)