#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

template <typename T>
__device__ __forceinline__ T reduceSumWarp(T sum) {
  unsigned int mask = __activemask();
  sum += __shfl_down_sync(mask, sum, 16);
  sum += __shfl_down_sync(mask, sum, 8);
  sum += __shfl_down_sync(mask, sum, 4);
  sum += __shfl_down_sync(mask, sum, 2);
  sum += __shfl_down_sync(mask, sum, 1);
  return sum;
}

// Kernel
template <typename T>
__global__ void reduceSum(const T *A, T *B, int count) {
  extern __shared__ T partialSums[];
  size_t tid = threadIdx.x;
  size_t bid = blockIdx.x;
  size_t wid = tid/32;
  size_t i = bid * blockDim.x + tid;


  // All able threads grab an element and reduce into warp leader
  T partialSum = reduceSumWarp(i < count ? A[i] : (T)0);
  // Leader stores into shared memory for inter-warp reduction
  if (tid % 32 == 0) {
    partialSums[wid] = partialSum;
  }
  __syncthreads();

  // First warp reduces previous warp sums
  // WARN: Only works if block contains more than one warp
  if (wid == 0) {
    int sharedCount = blockDim.x / 32;
    partialSum = reduceSumWarp(tid < sharedCount ? partialSums[tid] : (T)0);
    // First thread performs storage operation
    if (tid == 0) {
      B[bid] = partialSum;
    }
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
  constexpr int threadsPerBlock = 1024;
  constexpr int warpsPerBlock = threadsPerBlock/32;
  while (count > 1) {
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    reduceSum<<<blocksPerGrid, threadsPerBlock, warpsPerBlock * sizeof(float)>>>(dA, dB, count);
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
