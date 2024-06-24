#include "batched_spsc_queue.hh"
#include <chrono>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>

namespace dh {

class BatchedSPSCQueueMultiThreadingTest
    : public ::testing::TestWithParam<
          std::tuple<std::chrono::seconds, std::chrono::microseconds,
                     std::chrono::microseconds, size_t, size_t, size_t>> {};

TEST_P(BatchedSPSCQueueMultiThreadingTest, MT) {
  // Test parameters.
  auto [test_duration, enqueue_delay, dequeue_delay, nb_slots,
        enqueue_batch_size, dequeue_batch_size] = GetParam();

  // Create the queue.
  size_t element_size = sizeof(uint8_t);
  size_t buffer_size = nb_slots * element_size;
  std::vector<uint8_t> buffer(buffer_size);
  std::span<uint8_t> buffer_span(buffer);
  BatchedSPSCQueue queue(nb_slots, enqueue_batch_size, dequeue_batch_size,
                         element_size, buffer_span);

  // Enqueue thread.
  std::thread enqueue_thread(
      [&queue, test_duration, enqueue_batch_size, enqueue_delay]() {
        auto start_time = std::chrono::steady_clock::now();
        uint8_t data = 0;

        // Loop for the specified duration.
        while (std::chrono::steady_clock::now() - start_time < test_duration) {
          // Try to get a write pointer.
          auto write_span = queue.write_ptr();
          if (!write_span.has_value())
            continue;

          // Write data to the buffer.
          for (size_t j = 0; j < enqueue_batch_size; j++)
            write_span.value()[j] = data++;

          // Commit the write.
          queue.commit_write();

          // Sleep for the specified duration.
          while (std::chrono::steady_clock::now() - start_time < test_duration)
            std::this_thread::sleep_for(enqueue_delay);
        }
      });

  // Dequeue thread.
  std::thread dequeue_thread(
      [&queue, test_duration, dequeue_batch_size, dequeue_delay]() {
        auto start_time = std::chrono::steady_clock::now();
        uint8_t expected = 0;

        // Loop for the specified duration.
        while (std::chrono::steady_clock::now() - start_time < test_duration) {

          // Try to get a read pointer.
          auto read_span = queue.read_ptr();
          if (!read_span.has_value())
            continue;

          // Check the data.
          for (size_t j = 0; j < dequeue_batch_size; j++)
            ASSERT_EQ(read_span.value()[j], expected++);

          // Commit the read.
          queue.commit_read();

          // Sleep for the specified duration.
          while (std::chrono::steady_clock::now() - start_time < test_duration)
            std::this_thread::sleep_for(dequeue_delay);
        }
      });

  enqueue_thread.join();
  dequeue_thread.join();
}

INSTANTIATE_TEST_SUITE_P(
    BatchedSPSCQueueTestSuite, BatchedSPSCQueueMultiThreadingTest,
    ::testing::Values(
        // 00
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        2,                            // enqueue_batch_size
                        3),                           // dequeue_batch_size
        // 01
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(2), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        2,                            // enqueue_batch_size
                        3),                           // dequeue_batch_size
        // 02
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(1), // dequeue_delay
                        3000,                         // nb_slots
                        2,                            // enqueue_batch_size
                        3),                           // dequeue_batch_size
        // 03
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        3,                            // enqueue_batch_size
                        2),                           // dequeue_batch_size
        // 04
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(2), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        3,                            // enqueue_batch_size
                        2),                           // dequeue_batch_size
        // 05
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(1), // dequeue_delay
                        3000,                         // nb_slots
                        3,                            // enqueue_batch_size
                        2),                           // dequeue_batch_size
        // 06
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        10,                           // enqueue_batch_size
                        1000),                        // dequeue_batch_size
        // 07
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(2), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        10,                           // enqueue_batch_size
                        1000),                        // dequeue_batch_size
        // 08
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(1), // dequeue_delay
                        3000,                         // nb_slots
                        10,                           // enqueue_batch_size
                        1000),                        // dequeue_batch_size
        // 09
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        1000,                         // enqueue_batch_size
                        10),                          // dequeue_batch_size
        // 10
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(2), // enqueue_delay
                        std::chrono::microseconds(0), // dequeue_delay
                        3000,                         // nb_slots
                        1000,                         // enqueue_batch_size
                        10),                          // dequeue_batch_size
        // 11
        std::make_tuple(std::chrono::seconds(10),     // test_duration
                        std::chrono::microseconds(0), // enqueue_delay
                        std::chrono::microseconds(1), // dequeue_delay
                        3000,                         // nb_slots
                        1000,                         // enqueue_batch_size
                        10)));                        // dequeue_batch_size
} // namespace dh