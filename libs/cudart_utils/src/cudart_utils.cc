#include "cudart_utils.hh"
#include "cuda_runtime_api.h"
#include <cuda_runtime.h>
#include <system_error>

namespace dh {
const char *cudart_error_category::name() const noexcept { return "cudart"; }

std::string cudart_error_category::message(int ev) const {
  return cudaGetErrorString(static_cast<cudaError_t>(ev));
}

const std::error_category &cudart_category() {
  static cudart_error_category instance;
  return instance;
}

std::error_code make_error_code(cudaError_t e) {
  return {static_cast<int>(e), cudart_category()};
}

void DeviceDeleter::operator()(void *ptr) { CUDART_EXIT(cudaFree(ptr)); }

void HostDeleter::operator()(void *ptr) { CUDART_EXIT(cudaFreeHost(ptr)); }
} // namespace dh
