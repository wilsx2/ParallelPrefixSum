#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// Kernel
template <typename T>
__global__ void vector_add(const T *a, const T *b, T *c, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    c[i] = a[i] + b[i];
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
  std::vector<float> host_a(count);
  std::vector<float> host_b(count);
  std::vector<float> host_c(count);
  randomize_vector(host_a, -100.f, 100.f);
  randomize_vector(host_b, -100.f, 100.f);

  // Alloc device memory
  float *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, size);
  cudaMalloc(&dev_b, size);
  cudaMalloc(&dev_c, size);

  // Copy from CPU to GPU
  cudaMemcpy(dev_a, host_a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, host_b.data(), size, cudaMemcpyHostToDevice);

  // Call kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
  vector_add<float><<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, count);

  // Copy back to the CPU
  cudaMemcpy(host_c.data(), dev_c, size, cudaMemcpyDeviceToHost);

  // Print results
  auto print_num = [](auto num) { std::cout << num << ", "; };
  std::for_each(host_a.begin(), host_a.begin() + 5, print_num);
  std::cout << "... | A" << std::endl;
  std::for_each(host_b.begin(), host_b.begin() + 5, print_num);
  std::cout << "... | B" << std::endl;
  std::for_each(host_c.begin(), host_c.begin() + 5, print_num);
  std::cout << "... | C" << std::endl;

  // Free memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}
