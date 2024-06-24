#include "cudart_utils.hh"
#include <gtest/gtest.h>

namespace dh {

cudaError_t return_success() { return cudaSuccess; }

cudaError_t return_failure() { return cudaErrorMemoryAllocation; }

TEST(MacroTest, CudartExitMacroSuccess) { CUDART_EXIT(return_success()); }

TEST(MacroTest, CudartExitMacroFailure) {
  ASSERT_EXIT(CUDART_EXIT(return_failure()),
              ::testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(MacroTest, CudartExpectedMacroSuccess) {
  auto result = CUDART_EXPECTED(return_success());
  ASSERT_TRUE(result.has_value());
}

TEST(MacroTest, CudartExpectedMacroFailure) {
  auto result = CUDART_EXPECTED(return_failure());
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error().value(),
            static_cast<int>(cudaErrorMemoryAllocation));
}

TEST(MacroTest, CudartTryMacroSuccess) {
  auto testFunction = []() -> tl::expected<void, std::error_code> {
    CUDART_TRY(return_success());
    return {};
  };

  auto result = testFunction();
  ASSERT_TRUE(result.has_value());
}

TEST(MacroTest, CudartTryMacroFailure) {
  auto testFunction = []() -> tl::expected<void, std::error_code> {
    CUDART_TRY(return_failure());
    return {};
  };

  auto result = testFunction();
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error().value(),
            static_cast<int>(cudaErrorMemoryAllocation));
}

} // namespace dh