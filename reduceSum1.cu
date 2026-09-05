#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Kernel
template <typename T>
__global__ void reduceSum(const T *A, T *B) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  B[i] = A[i*2] + A[i*2+1];
  __syncthreads();
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
  int nextCount;
  for (; count >= 1; count = nextCount) {
    nextCount = count/2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (nextCount + threadsPerBlock - 1) / threadsPerBlock;
    reduceSum<<<blocksPerGrid, threadsPerBlock>>>(dA, dB);
    std::swap(dA, dB);
  }

  cudaMemcpy(hB.data(), dB, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Host Sum: " << hSum << std::endl;
  std::cout << "Device Sum: " << hB[0] << std::endl;
  

  // Free memory
  cudaFree(dA);
  cudaFree(dB);
}
