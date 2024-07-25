#include "cuaab.hh"
#include <cuda_runtime.h>

namespace dh {
struct cuaabContext {
  cudaStream_t stream;
};

const char *cuaabGetErrorString(cuaabStatus_t status) {
  switch (status) {
  case CUAAB_SUCCESS:
    return "cuaab API was successful";
  case CUAAB_INTERNAL_ERROR:
    return "driver or internal error";
  case CUAAB_INVALID_STRIDE:
    return "invalid stride";
  case CUAAB_INVALID_BATCH:
    return "invalid batch";
  default:
    return "unknown error";
  }
}

cuaabStatus_t cuaabCreate(cuaabHandle_t *handle) {
  *handle = new cuaabContext();
  if (!handle)
    return CUAAB_INTERNAL_ERROR;
  (*handle)->stream = 0;
  return CUAAB_SUCCESS;
}

cuaabStatus_t cuaabDestroy(cuaabHandle_t handle) {
  delete handle;
  return CUAAB_SUCCESS;
}

cuaabStatus_t cuaabSetStream(cuaabHandle_t handle, cudaStream_t stream) {
  handle->stream = stream;
  return CUAAB_SUCCESS;
}

/**
 * @brief AAB kernel
 *
 * This kernels compute the Average Along Batch (AAB) operation on the `input`
 * tensor and stores the result in the `output` tensor.
 */
__global__ void aab_kernel(const float *input, float *output, int stride,
                           int batch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= stride)
    return;

  float sum = 0.0f;
  for (int i = 0; i < batch; i++) {
    sum += input[i * stride + idx];
  }

  output[idx] = sum / batch;
}

cuaabStatus_t cuaabAAB(cuaabHandle_t handle, const float *input, float *output,
                       int stride, int batch) {
  if (stride <= 0)
    return CUAAB_INVALID_STRIDE;

  if (batch <= 0)
    return CUAAB_INVALID_BATCH;

  int num_blocks = (stride + 1023) / 1024;
  aab_kernel<<<num_blocks, 1024, 0, handle->stream>>>(input, output, stride,
                                                      batch);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return CUAAB_INTERNAL_ERROR;

  return CUAAB_SUCCESS;
}
} // namespace dh