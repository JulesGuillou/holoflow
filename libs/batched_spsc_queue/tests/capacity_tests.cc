#include "batched_spsc_queue.hh"
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>

namespace dh {
TEST(BATCHED_SPSC_QUEUE, Capacity_Is_Respected_100_1_1_1) {
  constexpr size_t NB_SLOTS = 100;
  constexpr size_t ENQUEUE_BATCH_SIZE = 1;
  constexpr size_t DEQUEUE_BATCH_SIZE = 1;
  constexpr size_t ELEMENT_SIZE = 1;
  constexpr size_t BUFFER_SIZE = NB_SLOTS * ELEMENT_SIZE;

  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);

  for (size_t i = 0; i < 100 * 10; i++) {
    BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                           ELEMENT_SIZE, buffer_span);

    // Enqueue-Dequeue i elements to shift internal read/write indexes by i.
    for (size_t j = 0; j < i; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be able to enqueue 99 elements.
    for (size_t j = 0; j < 99; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
    }

    // Should be full now.
    ASSERT_FALSE(queue.write_ptr().has_value());

    // Should be ablse to dequeue 99 elements.
    for (size_t j = 0; j < 99; j++) {
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be empty now.
    ASSERT_FALSE(queue.read_ptr().has_value());
  }
}

TEST(BATCHED_SPSC_QUEUE, Capacity_Is_Respected_300_3_2_1) {
  constexpr size_t NB_SLOTS = 300;
  constexpr size_t ENQUEUE_BATCH_SIZE = 3;
  constexpr size_t DEQUEUE_BATCH_SIZE = 2;
  constexpr size_t ELEMENT_SIZE = 1;
  constexpr size_t BUFFER_SIZE = NB_SLOTS * ELEMENT_SIZE;

  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);

  for (size_t i = 0; i < 300 * 10; i++) {
    BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                           ELEMENT_SIZE, buffer_span);

    // Enqueue-Dequeue i elements to shift internal read/write indexes by i.
    // In this case, one has to enqueue twice before being able to dequeue
    // everything (three times).
    for (size_t j = 0; j < i; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be able to enqueue 297 elements (99 3-elements enqueue).
    for (size_t j = 0; j < 99; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
    }

    // Should be full now.
    ASSERT_FALSE(queue.write_ptr().has_value());

    // Should be ablse to dequeue 298 elements (148 2-elements dequeue).
    for (size_t j = 0; j < 148; j++) {
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be empty now.
    ASSERT_FALSE(queue.read_ptr().has_value());
  }
}

TEST(BATCHED_SPSC_QUEUE, Capacity_Is_Respected_300_2_3_1) {
  constexpr size_t NB_SLOTS = 300;
  constexpr size_t ENQUEUE_BATCH_SIZE = 2;
  constexpr size_t DEQUEUE_BATCH_SIZE = 3;
  constexpr size_t ELEMENT_SIZE = 1;
  constexpr size_t BUFFER_SIZE = NB_SLOTS * ELEMENT_SIZE;

  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);

  for (size_t i = 0; i < 300 * 10; i++) {
    BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                           ELEMENT_SIZE, buffer_span);

    // Enqueue-Dequeue i elements to shift internal read/write indexes by i.
    // In this case, one has to enqueue three times before being able to dequeue
    // everything (twice).
    for (size_t j = 0; j < i; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be able to enqueue 298 elements (149 2-elements enqueue).
    for (size_t j = 0; j < 149; j++) {
      ASSERT_TRUE(queue.write_ptr().has_value());
      queue.commit_write();
    }

    // Should be full now.
    ASSERT_FALSE(queue.write_ptr().has_value());

    // Should be ablse to dequeue 297 elements (99 3-elements dequeue).
    for (size_t j = 0; j < 99; j++) {
      ASSERT_TRUE(queue.read_ptr().has_value());
      queue.commit_read();
    }

    // Should be empty now.
    ASSERT_FALSE(queue.read_ptr().has_value());
  }
}
} // namespace dh