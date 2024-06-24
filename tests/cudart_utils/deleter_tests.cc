#include "cudart_utils.hh"
#include <gtest/gtest.h>
#include <memory>

namespace dh {
TEST(DeleterTest, DeviceDeleterSuccess) {
  // Allocate some memory.
  void *ptr = nullptr;
  CUDART_EXIT(cudaMalloc(&ptr, 1024));

  // Check that the underlying data is accessible.
  CUDART_EXIT(cudaMemset(ptr, 0, 1024));

  // Delete the pointer.
  DeviceDeleter deleter;
  deleter(ptr);

  // Check that the underlying data is no longer accessible.
  ASSERT_EXIT(CUDART_EXIT(cudaMemset(ptr, 0, 1024)),
              ::testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(DeleterTest, DeviceDeleterFailure) {
  void *ptr = (void *)0xDEADBEEF;
  DeviceDeleter deleter;
  ASSERT_EXIT(deleter(ptr), ::testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(DeleterTest, HostDeleterSuccess) {
  // Allocate some memory.
  void *ptr =nullptr;
  CUDART_EXIT(cudaMallocHost(&ptr, 1024));

  // Check that the underlying data is accessible.
  CUDART_EXIT(cudaMemset(ptr, 0, 1024));

  // Delete the pointer.
  HostDeleter deleter;
  deleter(ptr);

  // Check that the underlying data is no longer accessible.
  ASSERT_EXIT(CUDART_EXIT(cudaMemset(ptr, 0, 1024)),
              ::testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(DeleterTest, HostDeleterFailure) {
  void *ptr = (void *)0xDEADBEEF;
  HostDeleter deleter;
  ASSERT_EXIT(deleter(ptr), ::testing::ExitedWithCode(EXIT_FAILURE), "");
}
} // namespace dh
