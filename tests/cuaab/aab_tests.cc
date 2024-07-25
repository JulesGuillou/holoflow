#include "cuaab.hh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>

namespace dh {
struct cuaabHandleDeleter {
  void operator()(cuaabHandle_t *handle) {
    auto status = cuaabDestroy(*handle);
    if (status != CUAAB_SUCCESS) {
      std::cerr << "Failed to destroy CUAAB handle: "
                << cuaabGetErrorString(status) << std::endl;
    }
  }
};

struct cudaMemoryDeleter {
  void operator()(float *ptr) {
    auto error = cudaFree(ptr);
    if (error != cudaSuccess) {
      std::cerr << "Failed to free CUDA memory: " << cudaGetErrorString(error)
                << std::endl;
    }
  }
};

TEST(AABTests, StrideIsZero) {
  // Create a new CUAAB library context.
  cuaabHandle_t handle_{};
  std::unique_ptr<cuaabHandle_t, cuaabHandleDeleter> handle(&handle_);
  cuaabStatus_t status = cuaabCreate(handle.get());
  ASSERT_EQ(status, CUAAB_SUCCESS)
      << "CUAAB error: " << cuaabGetErrorString(status);

  // Perform the AAB operation.
  constexpr int BATCH = 10;
  constexpr int STRIDE = 0;
  status = cuaabAAB(*handle, nullptr, nullptr, STRIDE, BATCH);
  ASSERT_EQ(status, CUAAB_INVALID_STRIDE);
}

TEST(AABTests, BatchIsZero) {
  // Create a new CUAAB library context.
  cuaabHandle_t handle_{};
  std::unique_ptr<cuaabHandle_t, cuaabHandleDeleter> handle(&handle_);
  cuaabStatus_t status = cuaabCreate(handle.get());
  ASSERT_EQ(status, CUAAB_SUCCESS)
      << "CUAAB error: " << cuaabGetErrorString(status);

  // Perform the AAB operation.
  constexpr int BATCH = 0;
  constexpr int STRIDE = 10;
  status = cuaabAAB(*handle, nullptr, nullptr, STRIDE, BATCH);
  ASSERT_EQ(status, CUAAB_INVALID_BATCH);
}

TEST(AABTests, Batch10Size10) {
  // Create a new CUAAB library context.
  cuaabHandle_t handle_{};
  std::unique_ptr<cuaabHandle_t, cuaabHandleDeleter> handle(&handle_);
  cuaabStatus_t status = cuaabCreate(handle.get());
  ASSERT_EQ(status, CUAAB_SUCCESS)
      << "CUAAB error: " << cuaabGetErrorString(status);

  // Prepare the input and output tensors.
  constexpr int BATCH = 10;
  constexpr int STRIDE = 10;

  float *input_ = nullptr;
  cudaError_t error = cudaMalloc(&input_, BATCH * STRIDE * sizeof(float));
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
  std::unique_ptr<float, cudaMemoryDeleter> input(input_);

  float *host_input_ = nullptr;
  error = cudaMallocHost(&host_input_, BATCH * STRIDE * sizeof(float));
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
  std::unique_ptr<float, cudaMemoryDeleter> host_input(host_input_);

  float *output_ = nullptr;
  error = cudaMalloc(&output_, STRIDE * sizeof(float));
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
  std::unique_ptr<float, cudaMemoryDeleter> output(output_);

  float *host_output_ = nullptr;
  error = cudaMallocHost(&host_output_, STRIDE * sizeof(float));
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);
  std::unique_ptr<float, cudaMemoryDeleter> host_output(host_output_);

  // Fill input tensor with [0, 1, 2, 3, ..., n] [n + 1, n + 2, ..., 2n] ...
  for (int i = 0; i < BATCH * STRIDE; i++) {
    host_input.get()[i] = static_cast<float>(i);
  }

  // Host to device memory copy.
  error = cudaMemcpy(input.get(), host_input.get(),
                     BATCH * STRIDE * sizeof(float), cudaMemcpyHostToDevice);
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

  // Perform the AAB operation.
  status = cuaabAAB(*handle, input.get(), output.get(), STRIDE, BATCH);
  ASSERT_EQ(status, CUAAB_SUCCESS);

  // Sync the stream and check for errors.
  cudaStreamSynchronize(nullptr);
  error = cudaGetLastError();
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

  // Device to host memory copy.
  error = cudaMemcpy(host_output.get(), output.get(), STRIDE * sizeof(float),
                     cudaMemcpyDeviceToHost);
  ASSERT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

  // Check the output tensor.
  for (int i = 0; i < STRIDE; i++) {
    // [0 * stride + i + 1 * stride + i + ... + (batch - 1) * stride + i] /
    // batch = [(0 + 1 + ... + batch - 1) * stride + batch * i] / batch = (batch
    // - 1) / 2 * stride + i
    float expected = (BATCH - 1) / 2.0f * STRIDE + static_cast<float>(i);
    ASSERT_FLOAT_EQ(host_output.get()[i], expected);
  }
}
} // namespace dh
