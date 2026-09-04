NVCC := nvcc
NVCCFLAGS := -O3 -arch=native -std=c++17

CU_SRCS := $(wildcard *.cu)
TARGETS := $(patsubst %.cu, %.out, $(CU_SRCS))

all: $(TARGETS)
%.out: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
