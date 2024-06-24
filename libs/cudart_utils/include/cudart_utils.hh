#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <system_error>
#include <tl/expected.hpp>

/**
 * @brief Macro to call a CUDA runtime function and exit on error.
 *
 * This macro calls a CUDA runtime function and checks for errors. If an error
 * occurs, it logs the error message and exits the program with status
 * EXIT_FAILURE.
 *
 * @param call The CUDA runtime function to call.
 */
#define CUDART_EXIT(call)                                                      \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA call `" << #call << "` failed at " << __FILE__ << ":" \
                << __LINE__ << " with error code = `" << err                   \
                << "` and message = `" << cudaGetErrorString(err) << "`"       \
                << std::endl;                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Macro to call a CUDA runtime function and return an expected result.
 *
 * This macro calls a CUDA runtime function and checks for errors. If an error
 * occurs, it returns a tl::unexpected with the corresponding std::error_code.
 * Otherwise, it returns an empty tl::expected.
 *
 * @param call The CUDA runtime function to call.
 * @return A tl::expected<void, std::error_code> indicating success or failure.
 */
#define CUDART_EXPECTED(call)                                                  \
  [&]() -> tl::expected<void, std::error_code> {                               \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      return tl::unexpected(dh::make_error_code(err));                 \
    }                                                                          \
    return {};                                                                 \
  }()

/**
 * @brief Macro to call a CUDA runtime function and return on error.
 *
 * This macro calls a CUDA runtime function and checks for errors. If an error
 * occurs, it returns a tl::unexpected with the corresponding std::error_code.
 *
 * @param call The CUDA runtime function to call.
 */
#define CUDART_TRY(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      return tl::unexpected(dh::make_error_code(err));                 \
    }                                                                          \
  } while (0)

namespace dh {
/**
 * @class cudart_error_category
 * @brief Custom error category for CUDA runtime errors.
 *
 * This class extends std::error_category to provide custom error messages
 * for CUDA runtime errors.
 */
class cudart_error_category : public std::error_category {
public:
  /**
   * @brief Returns the name of the error category.
   * @return A constant character pointer to the name of the error category.
   */
  const char *name() const noexcept override;

  /**
   * @brief Returns the error message corresponding to the error value.
   * @param ev The error value.
   * @return A string containing the error message.
   */
  std::string message(int ev) const override;
};

/**
 * @brief Returns a reference to the custom CUDA runtime error category.
 * @return A constant reference to the cudart_error_category instance.
 */
const std::error_category &cudart_category();

/**
 * @brief Creates an error code for a given CUDA runtime error.
 * @param e The CUDA runtime error.
 * @return A std::error_code object representing the CUDA runtime error.
 */
std::error_code make_error_code(cudaError_t e);

/**
 * @struct DeviceDeleter
 * @brief Custom deleter for CUDA device memory.
 *
 * This struct provides a custom deleter for use with smart pointers
 * managing CUDA device memory.
 */
struct DeviceDeleter {
  /**
   * @brief Deletes the CUDA device memory pointed to by ptr.
   *
   * The pointer must have been allocated by cudaMalloc. In case of failure,
   * the function will log the error and exit with status EXIT_FAILURE.
   *
   * @param ptr Pointer to the CUDA device memory to be deleted.
   */
  void operator()(void *ptr);
};

/**
 * @struct HostDeleter
 * @brief Custom deleter for CUDA host memory.
 *
 * This struct provides a custom deleter for use with smart pointers
 * managing CUDA host memory.
 */
struct HostDeleter {
  /**
   * @brief Deletes the CUDA host memory pointed to by ptr.
   *
   * The pointer must have been allocated by cudaMallocHost. In case of
   * failure, the function will log the error and exit with status
   * EXIT_FAILURE.
   *
   * @param ptr Pointer to the CUDA host memory to be deleted.
   */
  void operator()(void *ptr);
};

} // namespace dh
