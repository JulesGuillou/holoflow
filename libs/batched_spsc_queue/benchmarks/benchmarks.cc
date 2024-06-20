#include "batched_spsc_queue.hh"
#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

using BatchedSPSCQueue = dh::BatchedSPSCQueue;

constexpr size_t NB_SLOTS = 1000;
constexpr size_t ENQUEUE_BATCH_SIZE = 8;
constexpr size_t DEQUEUE_BATCH_SIZE = 8;
constexpr size_t ELEMENT_SIZE = 512 * 512 * sizeof(uint8_t);
constexpr size_t BUFFER_SIZE = NB_SLOTS * ELEMENT_SIZE;
constexpr size_t ENQUEUE_BYTES = ENQUEUE_BATCH_SIZE * ELEMENT_SIZE;
constexpr size_t DEQUEUE_BYTES = DEQUEUE_BATCH_SIZE * ELEMENT_SIZE;

static void BM_Enqueue_NoMemoryTransfer(benchmark::State &state) {
  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);
  BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                         ELEMENT_SIZE, buffer_span);

  for (auto _ : state) {
    auto batch = queue.write_ptr();
    if (!batch.has_value()) {
      queue.reset();
      batch = queue.write_ptr();
    }

    queue.commit_write();
  }

  state.counters["Enqueues"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

  state.counters["Bandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * ENQUEUE_BYTES,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

static void BM_Dequeue_NoMemoryTransfer(benchmark::State &state) {
  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);
  BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                         ELEMENT_SIZE, buffer_span);

  queue.fill();

  for (auto _ : state) {
    auto batch = queue.read_ptr();
    if (!batch.has_value()) {
      queue.fill();
      batch = queue.read_ptr();
    }

    queue.commit_read();
  }

  state.counters["Dequeues"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

  state.counters["Bandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * DEQUEUE_BYTES,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

static void BM_Enqueue_WithMemoryTransfer(benchmark::State &state) {
  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);
  BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                         ELEMENT_SIZE, buffer_span);

  std::array<uint8_t, ENQUEUE_BATCH_SIZE * ELEMENT_SIZE> source = {0};

  benchmark::DoNotOptimize(source);
  benchmark::DoNotOptimize(buffer.data());

  for (auto _ : state) {
    auto batch = queue.write_ptr();
    if (!batch.has_value()) {
      queue.reset();
      batch = queue.write_ptr();
    }

    std::copy(source.begin(), source.end(), batch.value().begin());
    queue.commit_write();
  }

  state.counters["Enqueues"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

  state.counters["Bandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * ENQUEUE_BYTES,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

static void BM_Dequeue_WithMemoryTransfer(benchmark::State &state) {
  std::vector<uint8_t> buffer(BUFFER_SIZE);
  std::span<uint8_t> buffer_span(buffer);
  BatchedSPSCQueue queue(NB_SLOTS, ENQUEUE_BATCH_SIZE, DEQUEUE_BATCH_SIZE,
                         ELEMENT_SIZE, buffer_span);

  std::array<uint8_t, DEQUEUE_BATCH_SIZE * ELEMENT_SIZE> dest = {0};
  queue.fill();

  benchmark::DoNotOptimize(dest);
  benchmark::DoNotOptimize(buffer.data());

  for (auto _ : state) {
    auto batch = queue.read_ptr();
    if (!batch.has_value()) {
      queue.fill();
      batch = queue.read_ptr();
    }

    std::copy(batch.value().begin(), batch.value().end(), dest.begin());
    queue.commit_read();
  }

  state.counters["Dequeues"] = benchmark::Counter(
      static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);

  state.counters["Bandwidth"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * DEQUEUE_BYTES,
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

// NOLINTBEGIN
BENCHMARK(BM_Enqueue_NoMemoryTransfer)->MinTime(5.0);
BENCHMARK(BM_Dequeue_NoMemoryTransfer)->MinTime(5.0);
BENCHMARK(BM_Enqueue_WithMemoryTransfer)->MinTime(5.0);
BENCHMARK(BM_Dequeue_WithMemoryTransfer)->MinTime(5.0);

BENCHMARK_MAIN();
// NOLINTEND