#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Kernel
template <typename T>
__global__ void reduceSum(const T *A, T *B, int count) {
  extern __shared__ T shared[];
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + tid;

  // Move data into shared memory with zero padding
  shared[tid] = (i < (size_t)count) ? A[i] : (T)0;
  __syncthreads();

  // Perform tree reduction
  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // Store reduction of blockDim elements into B
  if (tid == 0) {
    B[blockIdx.x] = shared[0];
  }
}

// Helper
// Assume floating point
template <typename T>
void randomizeVector(std::vector<T> &vec, T low, T high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dist(low, high);

  for (auto &val : vec) {
    val = dist(gen);
  }
}

int main() {
  // Alloc and init host memory
  std::size_t count = 1 << 16;
  std::size_t size = count * sizeof(float);
  std::vector<float> hA(count);
  std::vector<float> hB(count);
  randomizeVector(hA, -128.f, 128.f);
  float hSum = std::accumulate(hA.begin(), hA.end(), 0.f);

  // Alloc device memory
  float *dA;
  float *dB;
  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);

  // Copy from CPU to GPU
  cudaMemcpy(dA, hA.data(), size, cudaMemcpyHostToDevice);

  // Call kernel
  int threadsPerBlock = 256;
  while (count > 1) {
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    reduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(dA, dB, count);
    count = blocksPerGrid;
    if (count > 1) {
      std::swap(dA, dB);
    }
  }

  cudaMemcpy(hB.data(), dB, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Host Sum: " << hSum << std::endl;
  std::cout << "Device Sum: " << hB[0] << std::endl;

  // Free memory
  cudaFree(dA);
  cudaFree(dB);
}
