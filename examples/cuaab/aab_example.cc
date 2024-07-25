#include "cuaab.hh"
#include <iostream>

constexpr int BATCH = 10;
constexpr int STRIDE = 10;

int main() {
  // Create a new CUAAB library context.
  dh::cuaabHandle_t handle;
  dh::cuaabStatus_t status = dh::cuaabCreate(&handle);
  if (status != dh::CUAAB_SUCCESS) {
    std::cerr << "Failed to create CUAAB library context: "
              << dh::cuaabGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  // Use it here...
  float *input = nullptr;
  float *host_input = nullptr;
  float *output = nullptr;
  float *host_output = nullptr;
  cudaMalloc(&input, BATCH * STRIDE * sizeof(float));
  cudaMallocHost(&host_input, BATCH * STRIDE * sizeof(float));
  cudaMalloc(&output, STRIDE * sizeof(float));
  cudaMallocHost(&host_output, STRIDE * sizeof(float));

  // Fill input tensor with [0, 1, 2, 3, ..., n] [n + 1, n + 2, ..., 2n] ...
  for (int i = 0; i < BATCH * STRIDE; i++) {
    host_input[i] = static_cast<float>(i);
  }

  // Host to device memory copy.
  cudaMemcpy(input, host_input, BATCH * STRIDE * sizeof(float),
             cudaMemcpyHostToDevice);

  // Perform the AAB operation.
  status = cuaabAAB(handle, input, output, STRIDE, BATCH);
  if (status != dh::CUAAB_SUCCESS) {
    std::cerr << "Failed to perform AAB operation: "
              << dh::cuaabGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }

  // Device to host memory copy.
  cudaMemcpy(host_output, output, STRIDE * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Check the output tensor.
  for (int i = 0; i < STRIDE; i++) {
    // [0 * stride + i + 1 * stride + i + ... + (batch - 1) * stride + i] /
    // batch = [(0 + 1 + ... + batch - 1) * stride + batch * i] / batch = (batch
    // - 1) / 2 * stride + i
    float expected = (BATCH - 1) / 2.0f * STRIDE + static_cast<float>(i);
    if (host_output[i] != expected) {
      std::cerr << "Output tensor mismatch at index " << i << ": expected "
                << expected << ", got " << host_output[i] << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Destroy the CUAAB library context.
  status = dh::cuaabDestroy(handle);
  if (status != dh::CUAAB_SUCCESS) {
    std::cerr << "Failed to destroy CUAAB library context: "
              << dh::cuaabGetErrorString(status) << std::endl;
    return EXIT_FAILURE;
  }
}