#include "memvanta/prefetcher.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace memvanta {

Prefetcher::Prefetcher(const TensorStore& store, TensorCache& cache)
    : store_(store), cache_(cache) {
    worker_ = std::thread(&Prefetcher::loop, this);
}

Prefetcher::~Prefetcher() {
    stop();
}

void Prefetcher::rethrow_if_failed() {
    std::exception_ptr error;
    {
        std::lock_guard lk(mu_);
        error = error_;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void Prefetcher::request(std::uint32_t id) {
    rethrow_if_failed();
    if (stop_.load() || id >= store_.count() || cache_.contains(id)) {
        return;
    }

    const auto bytes = store_.slice(id).bytes;
    {
        std::lock_guard lk(mu_);
        if (stop_.load() || pending_.contains(id)) {
            return;
        }
        if (bytes > std::numeric_limits<std::uint64_t>::max() - pending_bytes_) {
            throw std::runtime_error("prefetch pending-byte accounting overflow");
        }
        pending_.emplace(id, bytes);
        pending_bytes_ += bytes;
        peak_pending_bytes_ = std::max(peak_pending_bytes_, pending_bytes_);
        q_.push_back(id);
    }
    cv_.notify_one();
}

bool Prefetcher::pending(std::uint32_t id) {
    std::lock_guard lk(mu_);
    return pending_.contains(id);
}

std::uint64_t Prefetcher::pending_bytes() {
    std::lock_guard lk(mu_);
    return pending_bytes_;
}

std::uint64_t Prefetcher::peak_pending_bytes() {
    std::lock_guard lk(mu_);
    return peak_pending_bytes_;
}

void Prefetcher::stop() {
    // The condition-variable predicate must be changed while holding the same
    // mutex used by wait(). Otherwise a stop notification can land between the
    // worker's predicate check and its transition to sleep, leaving join()
    // waiting forever on a worker that has already missed its final wake-up.
    {
        std::lock_guard lk(mu_);
        stop_.store(true);
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }

    // A stop cancels anything that was still queued. Keep peak telemetry, but do
    // not leave cancelled work visible as currently pending.
    {
        std::lock_guard lk(mu_);
        q_.clear();
        pending_.clear();
        pending_bytes_ = 0;
    }
}

void Prefetcher::loop() {
    while (true) {
        std::uint32_t id;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_.load() || !q_.empty(); });
            if (stop_.load()) {
                return;
            }
            id = q_.front();
            q_.pop_front();
        }

        try {
            store_.prefetch(id);
            const auto& slice = store_.slice(id);
            cache_.insert_prefetched(id, store_.ptr(id), slice.bytes);
            // Do not issue MADV_DONTNEED from the background worker. The consumer
            // releases the mapped source after it has copied/consumed the tensor.
            // This avoids concurrent WILLNEED/DONTNEED churn on the same mapping.
            std::lock_guard lk(mu_);
            if (auto it = pending_.find(id); it != pending_.end()) {
                pending_bytes_ -= it->second;
                pending_.erase(it);
            }
        } catch (...) {
            {
                std::lock_guard lk(mu_);
                if (!error_) {
                    error_ = std::current_exception();
                }
                q_.clear();
                pending_.clear();
                pending_bytes_ = 0;
                stop_.store(true);
            }
            cv_.notify_all();
            return;
        }
    }
}

} // namespace memvanta
