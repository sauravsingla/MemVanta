#pragma once
#include "memvanta/tensor_store.hpp"
#include "memvanta/lru_cache.hpp"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_set>
namespace memvanta {
class Prefetcher {
public:
  Prefetcher(const TensorStore& store, TensorCache& cache);
  ~Prefetcher();
  void request(std::uint32_t id);
  void stop();
  void rethrow_if_failed();
private:
  void loop();
  const TensorStore& store_; TensorCache& cache_;
  std::atomic<bool> stop_{false};
  std::mutex mu_; std::condition_variable cv_;
  std::deque<std::uint32_t> q_; std::unordered_set<std::uint32_t> pending_;
  std::exception_ptr error_;
  std::thread worker_;
};
}
