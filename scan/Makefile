EXECUTABLE := cudaScan

CU_FILES   := scan.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++
else
    CXX = g++ -m64
endif

OBJDIR=objs

# /usr/local/cuda on this devserver points to CUDA 12.8, and the CUDA 13
# compiler package is installed without runtime headers/libs. If the runtime
# dev RPM has been extracted locally, use it for headers and libcudart.
CUDA_HOME ?= /usr/local/cuda-13.0
CUDA_RUNTIME_HOME ?= $(firstword $(wildcard $(HOME)/.local/cuda-cudart-13.0/usr/local/cuda-13.0 $(CUDA_HOME)))
CUDA_INC_DIR ?= $(CUDA_RUNTIME_HOME)/include
CUDA_TARGET_INC_DIR ?= $(CUDA_RUNTIME_HOME)/targets/$(ARCH)-linux/include
CUDA_CCCL_INC_DIR ?= $(CUDA_TARGET_INC_DIR)/cccl
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_RUNTIME_HOME)/targets/$(ARCH)-linux/lib $(CUDA_RUNTIME_HOME)/lib64))

CXXFLAGS=-O3 -Wall -I$(CUDA_INC_DIR) -I$(CUDA_TARGET_INC_DIR) -I$(CUDA_CCCL_INC_DIR)
LDFLAGS=-L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR) -lcudart
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS=-O3 -m64 -arch=sm_80 -I$(CUDA_INC_DIR) -I$(CUDA_TARGET_INC_DIR) -I$(CUDA_CCCL_INC_DIR)


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/scan.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

check_scan: default
				./checker.py scan

check_find_repeats: default
				./checker.py find_repeats

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
