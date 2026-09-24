#pragma once

#include "memvanta/lru_cache.hpp"
#include "memvanta/tensor_store.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace memvanta {

class Prefetcher {
  public:
    Prefetcher(const TensorStore& store, TensorCache& cache);
    ~Prefetcher();

    void request(std::uint32_t id);
    bool pending(std::uint32_t id);
    std::uint64_t pending_bytes();
    std::uint64_t peak_pending_bytes();
    void stop();
    void rethrow_if_failed();

  private:
    void loop();

    const TensorStore& store_;
    TensorCache& cache_;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::uint32_t> q_;
    std::unordered_map<std::uint32_t, std::uint64_t> pending_;
    std::uint64_t pending_bytes_ = 0;
    std::uint64_t peak_pending_bytes_ = 0;
    std::exception_ptr error_;
    std::thread worker_;
};

} // namespace memvanta
