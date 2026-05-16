CONFIG  :=  ./config/Makefile.config

include $(CONFIG)


BUILD_PATH    :=  build
SAMPLE_PATH   :=  samples
SRC_PATH      :=  src
INC_PATH      :=  include
CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)

# Include source files from both samples and src directories
CXX_SRC       +=  $(wildcard $(SAMPLE_PATH)/*.cpp)
CXX_SRC       +=  $(wildcard $(SRC_PATH)/core/*.cpp)
CXX_SRC       +=  $(wildcard $(SRC_PATH)/tasks/*.cpp)
CXX_SRC       +=  $(wildcard $(SRC_PATH)/utils/*.cpp)
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/utils/*.cu)

APP_OBJS      :=  $(patsubst %.cpp, $(BUILD_PATH)/%.cpp.o, $(notdir $(CXX_SRC)))
APP_OBJS      +=  $(patsubst %.cu, $(BUILD_PATH)/%.cu.o, $(notdir $(KERNELS_SRC)))

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
                  -I $(INC_PATH) \
                  -I .

LIBS          :=  -L "$(CUDA_DIR)/lib64" \
                  -L "$(TENSORRT_INSTALL_DIR)/lib" \
                  -Wl,-rpath "$(CUDA_DIR)/lib64" \
                  -Wl,-rpath "$(TENSORRT_INSTALL_DIR)/lib" \
				  -lcudart -lcublas -lcudnn \
				  -lnvinfer -lnvonnxparser\
                  -lstdc++fs \
                  `pkg-config --libs opencv4`


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

update: $(APP)
	@echo finished updating $<

$(APP): $(APP_DEPS) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o bin/$@ $(LIBS) $(INCS)

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

.PHONY: all update show clean 