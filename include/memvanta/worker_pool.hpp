#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace memvanta {

class WorkerPool {
  public:
    explicit WorkerPool(unsigned threads = 1);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    unsigned size() const { return threads_; }
    void parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn);

  private:
    unsigned threads_{1};
    std::vector<std::thread> workers_;
    // A pool may be reused across many kernels, but only one generation may be
    // submitted at a time. Serializing callers prevents generation/completion state
    // from being overwritten by concurrent parallel_for invocations.
    std::mutex call_mu_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    std::function<void(std::size_t, std::size_t)> fn_;
    std::exception_ptr error_;
    std::size_t n_{0};
    std::size_t generation_{0};
    std::size_t finished_{0};
    bool stop_{false};
    void worker(unsigned tid);
};

} // namespace memvanta
