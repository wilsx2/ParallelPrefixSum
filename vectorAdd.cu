#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Kernel
template <typename T>
__global__ void vector_add(const T *A, const T *B, T *C, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// Helper
// Assume floating point
template <typename T>
void randomize_vector(std::vector<T> &vec, T low, T high) {
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
  std::vector<float> hC(count);
  randomize_vector(hA, -100.f, 100.f);
  randomize_vector(hB, -100.f, 100.f);

  // Alloc device memory
  float *dA, *dB, *dC;
  cudaMalloc(&dA, size);
  cudaMalloc(&dB, size);
  cudaMalloc(&dC, size);

  // Copy from CPU to GPU
  cudaMemcpy(dA, hA.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), size, cudaMemcpyHostToDevice);

  // Call kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
  vector_add<float><<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, count);

  // Copy back to the CPU
  cudaMemcpy(hC.data(), dC, size, cudaMemcpyDeviceToHost);

  // Print results
  auto print_num = [](auto num) { std::cout << num << ", "; };
  std::for_each(hA.begin(), hA.begin() + 5, print_num);
  std::cout << "... | A" << std::endl;
  std::for_each(hB.begin(), hB.begin() + 5, print_num);
  std::cout << "... | B" << std::endl;
  std::for_each(hC.begin(), hC.begin() + 5, print_num);
  std::cout << "... | C" << std::endl;

  // Free memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
