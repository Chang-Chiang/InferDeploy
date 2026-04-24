# Makefile for samples

# Compiler
CXX = g++
NVCC = nvcc

# CUDA and TensorRT paths
CUDA_VER = 12.2
CUDA_DIR = /usr/local/cuda-$(CUDA_VER)
TENSORRT_INSTALL_DIR = /home/cc/Downloads/TensorRT-8.6.1.6
OPENCV_INSTALL_DIR = /usr/local

# Includes
INCLUDES = \
    -I$(CUDA_DIR)/include \
    -I$(TENSORRT_INSTALL_DIR)/include \
    -I$(OPENCV_INSTALL_DIR)/include/opencv4 \
    -I../

# Libraries
LIBS = \
    -L$(CUDA_DIR)/lib64 \
    -L$(TENSORRT_INSTALL_DIR)/lib \
    -L../lib \
    -linfer_trt \
    -lnvinfer \
    -lnvonnxparser \
    -lcudnn \
    -lcublas \
    -lcuda \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lstdc++fs \
    -lpthread \
    -lcudart

# Flags
CXXFLAGS = -std=c++11 -Wall -Wextra -O3 $(INCLUDES)
NVCCFLAGS = -std=c++11 -O3 $(INCLUDES)

# Target
TARGET = ../bin/yolo_detector
SOURCE = yolo_detector.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) ../lib/libinfer_trt.a | ../bin
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

../bin:
	mkdir -p ../bin

../lib/libinfer_trt.a:
	@echo "Building main library first..."
	cd .. && mkdir -p build && cd build && cmake .. && make
	@echo "Main library built successfully."

clean:
	rm -f $(TARGET)

help:
	@echo "Usage:"
	@echo "  make              # Build the sample"
	@echo "  make clean        # Remove built executable"
	@echo ""
	@echo "Note: Ensure the main library (infer_trt.a) is built before building the sample."
	@echo "Run from project root: make all"
	@echo ""
	@echo "To run after building:"
	@echo "  ./$(TARGET) <path_to_onnx_model> <path_to_image>"