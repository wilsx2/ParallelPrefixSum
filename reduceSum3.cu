#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

template <int BlockSize, int MinBlockSize, typename T>
__device__ __forceinline__ void reduceSumShared(volatile T *shared, unsigned int tid) {
  if constexpr (BlockSize >= MinBlockSize) {
    if (tid < MinBlockSize/2)
      shared[tid] += shared[tid + MinBlockSize/2]; 

    constexpr int threadsPerWarp = 32;
    if constexpr (MinBlockSize/2 > threadsPerWarp)
      __syncthreads();
  }
}
// Kernel
template <int BlockSize, typename T>
__global__ void reduceSum(const T *A, T *B, int count) {
  extern __shared__ T shared[];
  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + tid;

  // Move data into shared memory with zero padding
  shared[tid] = (i < (size_t)count) ? A[i] : (T)0; // TODO: Perform first reduction on load
  __syncthreads();

  // Perform unrolled tree reduction
  reduceSumShared<BlockSize, 1024 >(shared, tid);
  reduceSumShared<BlockSize, 512  >(shared, tid);
  reduceSumShared<BlockSize, 256  >(shared, tid);
  reduceSumShared<BlockSize, 128  >(shared, tid);
  reduceSumShared<BlockSize, 64   >(shared, tid);
  reduceSumShared<BlockSize, 32   >(shared, tid);
  reduceSumShared<BlockSize, 16   >(shared, tid);
  reduceSumShared<BlockSize, 8    >(shared, tid);
  reduceSumShared<BlockSize, 4    >(shared, tid);
  reduceSumShared<BlockSize, 2    >(shared, tid);

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
  constexpr int threadsPerBlock = 256;
  while (count > 1) {
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    reduceSum<threadsPerBlock><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(dA, dB, count);
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
