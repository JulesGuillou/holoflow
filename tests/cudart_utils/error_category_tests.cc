#include "cudart_utils.hh"
#include <gtest/gtest.h>

namespace dh {

TEST(ErrorCategoryTest, CudartErrorCategoryName) {
  dh::cudart_error_category category;
  ASSERT_STREQ(category.name(), "cudart");
}

TEST(ErrorCategoryTest, CudartErrorCategoryMessage) {
  dh::cudart_error_category category;
  ASSERT_EQ(category.message(cudaErrorMemoryAllocation), "out of memory");
}

TEST(ErrorCategoryTest, CudartCategoryInstance) {
  const std::error_category &category = dh::cudart_category();
  ASSERT_STREQ(category.name(), "cudart");
}

TEST(ErrorCategoryTest, MakeErrorCode) {
  std::error_code ec = dh::make_error_code(cudaErrorMemoryAllocation);
  ASSERT_EQ(ec.value(), static_cast<int>(cudaErrorMemoryAllocation));
  ASSERT_STREQ(ec.category().name(), "cudart");
}

} // namespace dh