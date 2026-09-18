CONFIG  :=  ./config/Makefile.config

include $(CONFIG)


BUILD_PATH    :=  build
SAMPLE_PATH   :=  samples
SRC_PATH      :=  src
INC_PATH      :=  include
CUDA_DIR      :=  /usr

# Include source files from both samples and src directories
EVAL_MAIN     :=  $(SAMPLE_PATH)/quant_eval.cpp
VOL_MAIN      :=  $(SAMPLE_PATH)/volume_seg.cpp
CXX_SRC       +=  $(filter-out $(EVAL_MAIN) $(VOL_MAIN), $(wildcard $(SAMPLE_PATH)/*.cpp))
# OpenVINO 后端：OvBackend.cpp 仅在 ENABLE_OPENVINO=1 时参与编译（wildcard 中需排除，避免重复）
CXX_SRC       +=  $(filter-out $(SRC_PATH)/core/OvBackend.cpp, $(wildcard $(SRC_PATH)/core/*.cpp))
CXX_SRC       +=  $(wildcard $(SRC_PATH)/tasks/*.cpp)
CXX_SRC       +=  $(wildcard $(SRC_PATH)/utils/*.cpp)
ifeq ($(ENABLE_OPENVINO),1)
CXX_SRC       +=  $(SRC_PATH)/core/OvBackend.cpp
endif
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/utils/*.cu)

APP_OBJS      :=  $(patsubst %.cpp, $(BUILD_PATH)/%.cpp.o, $(notdir $(CXX_SRC)))
APP_OBJS      +=  $(patsubst %.cu, $(BUILD_PATH)/%.cu.o, $(notdir $(KERNELS_SRC)))

# 量化精度评估工具：主程序源去掉 sample.o，再加上 quant_eval.o
EVAL_APP      :=  trt-quant-eval
EVAL_OBJS     :=  $(filter-out $(BUILD_PATH)/sample.cpp.o, $(APP_OBJS)) $(BUILD_PATH)/quant_eval.cpp.o

# 体数据逐切面分割工具：主程序源去掉 sample.o，再加上 volume_seg.o
VOL_APP       :=  trt-volume-seg
VOL_OBJS      :=  $(filter-out $(BUILD_PATH)/sample.cpp.o, $(APP_OBJS)) $(BUILD_PATH)/volume_seg.cpp.o

APP_MKS       :=  $(APP_OBJS:.o=.mk)

APP_DEPS      :=  $(CXX_SRC)
APP_DEPS      +=  $(KERNELS_SRC)
APP_DEPS      +=  $(wildcard $(INC_PATH)/**/*.h)


CUCC          :=  $(CUDA_DIR)/bin/nvcc
CXXFLAGS      :=  -std=c++17 -pthread -fPIC
CUDAFLAGS     :=  --shared -Xcompiler -fPIC 

INCS          :=  -I $(CUDA_DIR)/include \
                  -I $(OPENCV_INSTALL_DIR) \
                  -I $(TENSORRT_INSTALL_DIR)/include \
                  -I $(TENSORRT_INSTALL_DIR)/include/x86_64-linux-gnu \
                  -I $(INC_PATH) \
                  -I .

LIBS          :=  -L "$(CUDA_DIR)/lib/x86_64-linux-gnu" \
                  -L "$(TENSORRT_INSTALL_DIR)/lib/x86_64-linux-gnu" \
                  -Wl,-rpath "$(CUDA_DIR)/lib/x86_64-linux-gnu" \
                  -Wl,-rpath "$(TENSORRT_INSTALL_DIR)/lib/x86_64-linux-gnu" \
                  -Wl,--no-as-needed \
				  -lcudart -lcublas \
				  -lnvinfer -lnvinfer_plugin -lnvonnxparser\
                  -lstdc++fs \
                  `pkg-config --libs opencv4`

# OpenVINO 后端（条件编译：config/Makefile.config 中 ENABLE_OPENVINO=1 时启用）
ifeq ($(ENABLE_OPENVINO),1)
CXXFLAGS      +=  -DENABLE_OPENVINO
INCS          +=  -I $(OPENVINO_INSTALL_DIR)/include
LIBS          +=  -L "$(OPENVINO_INSTALL_DIR)/lib" \
                  -Wl,-rpath "$(OPENVINO_INSTALL_DIR)/lib" \
                  -lopenvino
endif


ifeq ($(DEBUG),1)
CUDAFLAGS     +=  -g -O0
CXXFLAGS      +=  -g -O0
else
CUDAFLAGS     +=  -O3
CXXFLAGS      +=  -O3
endif

ifeq ($(SHOW_WARNING),1)
CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
else
CUDAFLAGS     +=  -w
CXXFLAGS      +=  -w
endif


ifeq (, $(shell which bear))
BEARCMD       :=
else
ifeq (bear 3.0.18, $(shell bear --version))
BEARCMD       := bear --output config/compile_commands.json --
else
BEARCMD       := bear -o config/compile_commands.json
endif
endif


all: 
	@mkdir -p bin
	@$(BEARCMD) $(MAKE) --no-print-directory $(APP)
	@echo finished building $@. Have fun!!

run:
	@$(MAKE) --no-print-directory update
	@./bin/$(APP)

quant-eval: $(EVAL_APP)
	@echo finished updating $<

volume-seg: $(VOL_APP)
	@echo finished updating $<

# 3D 后处理流水线（post3d/ 子模块，独立 nvcc 构建；TMPDIR 指向工作区避免沙箱限制）
liver3d:
	@mkdir -p post3d/nvcc_tmp
	@$(MAKE) -C post3d TMPDIR=$(CURDIR)/post3d/nvcc_tmp
	@cp -f post3d/bin/liver3d bin/liver3d

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o bin/$@ $(LIBS) $(INCS)

$(EVAL_APP): $(APP_DEPS) $(EVAL_OBJS)
	@$(CXX) $(EVAL_OBJS) -o bin/$@ $(LIBS) $(INCS)

$(VOL_APP): $(APP_DEPS) $(VOL_OBJS)
	@$(CXX) $(VOL_OBJS) -o bin/$@ $(LIBS) $(INCS)

# Add compilation rules for object files
$(BUILD_PATH)/%.cpp.o: $(SAMPLE_PATH)/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/core/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/tasks/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

$(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/utils/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/utils/%.cu
	@mkdir -p $(dir $@)
	@$(CUCC) $(CUDAFLAGS) $(INCS) -c $< -o $@

show: 
	@echo $(BUILD_PATH)
	@echo $(APP_DEPS)
	@echo $(INCS)
	@echo $(APP_OBJS)
	@echo $(APP_MKS)

clean:
	rm -rf $(APP)
	rm -rf build
	rm -rf config/compile_commands.json
	rm -rf bin

ifneq ($(MAKECMDGOALS), clean)
-include $(APP_MKS)
endif

# Compile CXX
$(BUILD_PATH)/%.cpp.o: %.cpp 
	@echo Compile CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCS)
$(BUILD_PATH)/%.cpp.mk: %.cpp
	@echo Compile Dependence CXX $@
	@mkdir -p $(BUILD_PATH)
	@$(CXX) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(CXXFLAGS) $(INCS) 

# Compile CUDA
$(BUILD_PATH)/%.cu.o: %.cu
	@echo Compile CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)
$(BUILD_PATH)/%.cu.mk: %.cu
	@echo Compile Dependence CUDA $@
	@mkdir -p $(BUILD_PATH)
	@$(CUCC) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(CUDAFLAGS)

.PHONY: all update quant-eval volume-seg liver3d show clean 