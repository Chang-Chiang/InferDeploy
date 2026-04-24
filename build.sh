#!/bin/bash

# Build script for InferDeploy project

set -e

# Configuration
BUILD_DIR=build
LIB_DIR=lib
BIN_DIR=bin

# Clean previous build
echo "Cleaning previous build..."
rm -rf ${BUILD_DIR} ${LIB_DIR} ${BIN_DIR}

# Create directories
mkdir -p ${BUILD_DIR} ${LIB_DIR} ${BIN_DIR}

# Configure with CMake
echo "Configuring project with CMake..."
cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_VER=12.2 \
    -DCUDA_DIR=/usr/local/cuda-12.2 \
    -DTENSORRT_INSTALL_DIR=/home/cc/Downloads/TensorRT-8.6.1.6 \
    -DOPENCV_INSTALL_DIR=/usr/local

# Build the library
echo "Building infer_trt library..."
cmake --build . --target infer_trt -j$(nproc)

# Build the sample executable
echo "Building yolo_detector sample..."
cmake --build . --target yolo_detector -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "Library:   ${LIB_DIR}/libInferDeploy.a"
echo "Executable: ${BIN_DIR}/yolo_detector"
echo "=========================================="