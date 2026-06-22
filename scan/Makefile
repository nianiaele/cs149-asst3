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

# /usr/local/cuda on this devserver points to a stripped install missing
# headers/libs. Use the full CUDA 13 install instead.
CUDA_HOME ?= /usr/local/cuda-13

CXXFLAGS=-O3 -Wall -I$(CUDA_HOME)/include
LDFLAGS=-L$(CUDA_HOME)/lib64/ -lcudart
NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS=-O3 -m64 -arch=sm_80


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
