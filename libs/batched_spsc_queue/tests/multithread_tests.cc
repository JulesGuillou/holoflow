#include "batched_spsc_queue.hh"
#include <chrono>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>

constexpr size_t NB_SLOTS = 3000;
constexpr size_t ELEMENT_SIZE = sizeof(uint8_t);
constexpr size_t BUFFER_SIZE = NB_SLOTS * ELEMENT_SIZE;
constexpr std::chrono::seconds TEST_DURATION(10);

#define GENERATE_TEST_CASE(TEST_SUITE_NAME, TEST_NAME, ENQUEUE_DELAY_US,       \
                           DEQUEUE_DELAY_US, ENQUEUE_BATCH_SIZE,               \
                           DEQUEUE_BATCH_SIZE)                                 \
  TEST(TEST_SUITE_NAME, TEST_NAME) {                                           \
    std::vector<uint8_t> buffer(BUFFER_SIZE);                                  \
    std::span<uint8_t> buffer_span(buffer);                                    \
    BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,   \
                           ELEMENT_SIZE, buffer_span);                         \
    std::future<bool> enqueue_thread =                                         \
        std::async(enqueue_task, std::ref(queue), TEST_DURATION,               \
                   ENQUEUE_BATCH_SIZE, ENQUEUE_DELAY_US);                      \
    std::future<bool> dequeue_thread =                                         \
        std::async(dequeue_task, std::ref(queue), TEST_DURATION,               \
                   DEQUEUE_BATCH_SIZE, DEQUEUE_DELAY_US);                      \
    EXPECT_TRUE(dequeue_thread.get());                                         \
    EXPECT_TRUE(enqueue_thread.get());                                         \
  }

namespace dh {
void sleep_us(size_t us) {
  auto start = std::chrono::high_resolution_clock::now();
  auto end = start + std::chrono::microseconds(us);

  while (std::chrono::high_resolution_clock::now() < end) {
  }
}

bool enqueue_task(BatchedSPSCQueue &queue, std::chrono::seconds test_duration,
                  size_t enqueue_batch_size, size_t wait_us) {
  auto start_time = std::chrono::steady_clock::now();
  uint8_t data = 0;

  while (std::chrono::steady_clock::now() - start_time < test_duration) {
    auto write_span = queue.write_ptr();
    if (!write_span.has_value())
      continue;

    for (size_t j = 0; j < enqueue_batch_size; j++)
      write_span.value()[j] = data++;

    queue.commit_write();
    sleep_us(wait_us);
  }

  return true;
}

bool dequeue_task(BatchedSPSCQueue &queue, std::chrono::seconds test_duration,
                  size_t dequeue_batch_size, size_t wait_us) {
  auto start_time = std::chrono::steady_clock::now();
  uint8_t expected = 0;

  while (std::chrono::steady_clock::now() - start_time < test_duration) {
    auto read_span = queue.read_ptr();
    if (!read_span.has_value())
      continue;

    for (size_t j = 0; j < dequeue_batch_size; j++)
      if (read_span.value()[j] != expected++)
        throw std::runtime_error("dequeue_task invalid data.");

    queue.commit_read();
    sleep_us(wait_us);
  }

  return true;
}

GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_00, 0, 0, 2, 3)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_01, 2, 0, 2, 3)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_02, 0, 1, 2, 3)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_03, 0, 0, 3, 2)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_04, 2, 0, 3, 2)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_05, 0, 1, 3, 2)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_06, 0, 0, 10, 1000)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_07, 2, 0, 10, 1000)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_08, 0, 1, 10, 1000)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_09, 0, 0, 1000, 10)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_10, 2, 0, 1000, 10)
GENERATE_TEST_CASE(BATCHED_SPSC_QUEUE, MT_11, 0, 1, 1000, 10)
} // namespace dh