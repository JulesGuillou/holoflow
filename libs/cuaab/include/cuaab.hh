#pragma once

#include <cuda_runtime.h>

namespace dh {
/**
 * @brief Status codes for the CUAAB library.
 *
 * All cuaab functions return a status code of this type.
 */
typedef enum cuaabStatus_t {
  CUAAB_SUCCESS = 0,    /**< The operation was successful. */
  CUAAB_INTERNAL_ERROR, /**< Driver or internal error occurred. */
  CUAAB_INVALID_STRIDE, /**< Stride is invalid. */
  CUAAB_INVALID_BATCH,  /**< Batch is invalid. */
} cuaabStatus_t;

/**
 * @brief Opaque structure holding CUAAB library context.
 *
 * This structure is not defined in the header file in order to hide the
 * implementation details from the user.
 */
struct cuaabContext;

/**
 * @brief Handle to the CUAAB library context.
 *
 * This handle is used to refer to the CUAAB library context in all CUAAB
 * functions. The context is created by `cuaabCreate()` and destroyed by
 * `cuaabDestroy()`.
 */
typedef cuaabContext *cuaabHandle_t;

/**
 * @brief Get the error string for a CUAAB status code.
 *
 * @return A C-style string describing the error.
 */
const char *cuaabGetErrorString(cuaabStatus_t status);

/**
 * @brief Create a new CUAAB library context.
 *
 * This function creates a new CUAAB library context and returns a handle to it.
 *
 * @param handle Pointer to the handle to the new CUAAB library context.
 *
 * @return A cuaabStatus_t status code indicating the result of the operation.
 * The possible status codes are:
 * - `CUAAB_SUCCESS` if no errors happened.
 * - `CUAAB_INTERNAL_ERROR` if the call to malloc to allocate memory for the
 * cuaabContext fails.
 *
 * The following example demonstrates how to use this function to create a new
 * CUAAB library context:
 * @include examples/cuaab/handle_create_destroy_example.cc
 *
 * @note If the call to this function fails, one should not call `cuaabDestroy`
 * on the handle.
 *
 * @warning This function will lead to undefined behaviour if the handle pointer
 * points to an invalid handle location. It must be non-null and the memory it
 * points to muist be big enough to hold a `cuAABHandle_t`.
 */
cuaabStatus_t cuaabCreate(cuaabHandle_t *handle);

/**
 * @brief Destroy a CUAAB library context.
 *
 * This function destroys a CUAAB library context and frees all resources
 * associated with it.
 *
 * @param handle Handle to the CUAAB library context to destroy.
 *
 * @return A cuaabStatus_t status code indicating the result of the operation.
 * The possible status codes are:
 * - `CUAAB_SUCCESS` if no errors happened.
 *
 * The following example demonstrates how to use this function to destroy a
 * CUAAB library context:
 * @include examples/cuaab/handle_create_destroy_example.cc
 *
 * @note A `nullptr` is considered as a valid entry for this function. In which
 * case the function will do nothing.
 *
 * @warning This function will lead to undefined behaviour if the handle is not
 * properly initialized.
 */
cuaabStatus_t cuaabDestroy(cuaabHandle_t handle);

/**
 * @brief Set the stream for the given handle.
 *
 * This function sets the `cudaStream_t` to use for operations on the given
 * `cuaabHandle_t`.
 *
 * @param handle Handle to the CUAAB library context.
 * @param stream CUDA stream to use for operations.
 *
 * @return A cuaabStatus_t status code indicating the result of the operation.
 * The possible status codes are:
 * - `CUAAB_SUCCESS` if no errors happened.
 *
 * @warning This function will lead to undefined behaviour if the following
 * conditions are not met:
 * - The handle is properly initialized.
 * - The handle is not being used in any other operation.
 * - The stream is a valid CUDA stream.
 */
cuaabStatus_t cuaabSetStream(cuaabHandle_t handle, cudaStream_t stream);

/**
 * @brief Perform a CUAAB operation.
 *
 * This function will perform Average Along Batch (AAB) operation on the input
 * tensor and return the result in the output tensor.
 *
 * @param handle Handle to the CUAAB library context.
 * @param input Pointer to the input data.
 * @param output Pointer to the output data.
 * @param stride The distance in elements between the same position in
 * consecutive batches.
 * @param batch The number of batches.
 *
 * @return A cuaabStatus_t status code indicating the result of the operation.
 * The possible status codes are:
 * - `CUAAB_SUCCESS` if no errors happened.
 * - `CUAAB_INVALID_STRIDE` if the stride is invalid (less or equal to zero).
 * - `CUAAB_INVALID_BATCH` if the batch is invalid (less or equal to zero).
 * - `CUAAB_INTERNAL_ERROR` if a driver or internal error occurred.
 * 
 * The following example demonstrates how to use this function to perform an AAB
 * operation on 10 vectors of size 10 (or 10 matrices of size 5x2, etc...):
 * @include examples/cuaab/aab_example.cc
 *
 * @warning This function will lead to undefined behaviour if the follwing
 * conditions are not met:
 * - The handle is properly initialized.
 * - The input must be a valid pointer to a memory location with enough space to
 * hold `stride * batch` elements.
 * - The output must be a valid pointer to a memory location with enough space
 * to hold `stride` elements.
 */
cuaabStatus_t cuaabAAB(cuaabHandle_t handle, const float *input, float *output,
                       int stride, int batch);
} // namespace dh