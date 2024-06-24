#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 128
#endif

namespace dh {
/**
 * @class BatchedSPSCQueue
 * @brief A batched single-producer single-consumer (SPSC) queue implemented
 * using a circular buffer.
 *
 * This class provides a high-performance queue that allows batching of enqueue
 * and dequeue operations. It is designed for scenarios where a single producer
 * thread enqueues data and a single consumer thread dequeues data. The queue
 * uses a pre-allocated memory buffer to store elements, and it supports batched
 * operations to improve performance. Sequential elements in a batch are
 * guaranteed to be sequential in memory.
 *
 * @note Not meeting the specified conditions results in undefined behavior.
 *       This includes:
 *       - Using the queue with multiple threads for enqueuing.
 *       - Using the queue with multiple threads for dequeuing.
 *       - Writing more than enqueue_batch_size elements in a single batch.
 *       - Writing less than dequeue_batch_size elements in a single batch.
 *       - Reading more than dequeue_batch_size elements in a single batch.
 *       - Reading less than enqueue_batch_size elements in a single batch.
 */
class BatchedSPSCQueue {
public:
  /**
   * @brief Constructs a BatchedSPSCQueue with the specified parameters.
   *
   * @param nb_slots The number of slots in the circular buffer. Note that not
   * all slots can be used simultaneously. The actual capacity of the queue is
   * nb_slots - enqueue_batch_size to avoid ambiguity between full and empty
   * states. Additionally, nb_slots must be a multiple of both
   * enqueue_batch_size and dequeue_batch_size.
   * @param enqueue_batch_size The number of elements that can be
   * enqueued in a single batch.
   * @param dequeue_batch_size The number of elements that can be
   * dequeued in a single batch.
   * @param element_size The size of each element in bytes.
   * @param buffer A pre-allocated memory block that is large enough to contain
   * nb_slots * element_size bytes.
   *
   * @note Not meeting the specified conditions results in undefined behavior.
   *       This includes:
   *       - nb_slots not being a multiple of enqueue_batch_size or
   * dequeue_batch_size.
   *      - buffer not being large enough to contain nb_slots * element_size
   */
  BatchedSPSCQueue(size_t nb_slots, size_t enqueue_batch_size,
                   size_t dequeue_batch_size, size_t element_size,
                   std::span<uint8_t> buffer);

  /**
   * @brief Returns a span containing the buffer where the user can write data.
   *
   * This method provides a span where the next element(s) can be written. The
   * caller is responsible for ensuring that the write does not exceed the
   * available space. If the queue is full, this method returns std::nullopt.
   *
   * @return A span containing the buffer where the user can write data, or
   * std::nullopt if the queue is full.
   *
   * @note The span returned by this method is invalidated after calling
   * commit_write().
   */
  std::optional<std::span<uint8_t>> write_ptr();

  /**
   * @brief Commits the write operation.
   *
   * This method updates the write index after writing data to the buffer. It
   * should be called after writing data to the span returned by
   * write_ptr().
   *
   * @note The span returned by write_ptr() is invalidated after calling this
   * method.
   */
  void commit_write();

  /**
   * @brief Returns a span containing the buffer where the user can read data.
   *
   * This method provides a span where the next element(s) can be read. The
   * caller is responsible for ensuring that the read does not exceed the
   * available space. If the queue is empty, this method returns std::nullopt.
   *
   * @return A span containing the buffer where the user can read data, or
   * std::nullopt if the queue is empty.
   *
   * @note The span returned by this method is invalidated after calling
   * commit_read().
   */
  std::optional<std::span<uint8_t>> read_ptr();

  /**
   * @brief Commits the read operation.
   *
   * This method updates the read index after reading data from the buffer. It
   * should be called after reading data from the span returned by
   * read_ptr().
   *
   * @note The span returned by read_ptr() is invalidated after calling this
   * method.
   */
  void commit_read();

  /**
   * @brief Returns the number of elements currently in the queue.
   *
   * This method provides the current size of the queue, i.e., the number of
   * elements that have been enqueued but not yet dequeued.
   *
   * @return The number of elements currently in the queue.
   */
  size_t size();

  /**
   * @brief Resets the queue.
   *
   * This method clears the queue and resets the read and write indices. It is
   * not thread-safe and should only be used for testing or benchmarking
   * purposes.
   */
  void reset();

  /**
   * @brief Fills the queue with uninitialized data.
   *
   * This method fills the queue with uninitialized data. It is not thread-safe
   * and should only be used for testing or benchmarking purposes.
   */
  void fill();

private:
  /**
   * @brief Returns the number of elements currently in the queue.
   *
   * This method provides the current size of the queue, i.e., the number of
   * elements that have been enqueued but not yet dequeued.
   *
   * @return The number of elements currently in the queue.
   *
   * @note This method should only be called by the producer thread.
   */
  size_t writer_size();

  /**
   * @brief Returns the number of elements currently in the queue.
   *
   * This method provides the current size of the queue, i.e., the number of
   * elements that have been enqueued but not yet dequeued.
   *
   * @return The number of elements currently in the queue.
   *
   * @note This method should only be called by the consumer thread.
   */
  size_t reader_size();

private:
  /// The number of slots in the circular buffer.
  size_t nb_slots_;

  /// The maximum number of elements that can be enqueued in a single batch.
  size_t enqueue_batch_size_;

  /// The maximum number of elements that can be dequeued in a single batch.
  size_t dequeue_batch_size_;

  /// The size of each element in bytes.
  size_t element_size_;

  /// A pre-allocated memory block for storing elements.
  std::span<uint8_t> buffer_;

  /// The current write index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_idx_;

  /// The current read index.
  alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_idx_;
};

} // namespace dh