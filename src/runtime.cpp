#include "memvanta/runtime.hpp"
#include "memvanta/common.hpp"
#include "memvanta/prefetch_policy.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <sys/resource.h>
#include <unordered_map>

namespace memvanta {

Runtime::Runtime(const TensorStore& s, RunConfig c)
    : store_(s), cfg_(c), cache_(c.cache_bytes), prefetcher_(s, cache_) {
    if (cfg_.copy_cache && cfg_.prefetch_budget_bytes > cfg_.cache_bytes) {
        cfg_.prefetch_budget_bytes = cfg_.cache_bytes;
    }
    cfg_.adaptive_low_useful_ratio = std::clamp(cfg_.adaptive_low_useful_ratio, 0.0, 1.0);
    cfg_.adaptive_high_useful_ratio =
        std::clamp(cfg_.adaptive_high_useful_ratio, cfg_.adaptive_low_useful_ratio, 1.0);
    if (cfg_.adaptive_prefetch) {
        if (cfg_.adaptive_min_depth == 0) cfg_.adaptive_min_depth = 1;
        if (cfg_.adaptive_max_depth < cfg_.adaptive_min_depth) {
            cfg_.adaptive_max_depth = cfg_.adaptive_min_depth;
        }
        if (cfg_.adaptive_window == 0) cfg_.adaptive_window = 1;
        cfg_.prefetch_depth =
            std::clamp(cfg_.prefetch_depth, cfg_.adaptive_min_depth, cfg_.adaptive_max_depth);
    }
}

std::uint64_t Runtime::rss_kb() {
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(r.ru_maxrss / 1024);
#else
    return static_cast<std::uint64_t>(r.ru_maxrss);
#endif
}

RunStats Runtime::run_stream() {
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t checksum = 1469598103934665603ull;
    std::uint64_t total = 0;
    std::uint32_t depth = cfg_.prefetch_depth;
    PrefetchStats pf{};

    const std::uint64_t unlimited = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t configured_budget =
        cfg_.prefetch_budget_bytes
            ? cfg_.prefetch_budget_bytes
            : (cfg_.copy_cache ? cfg_.cache_bytes : unlimited);

    // Copy-cache adaptive look-ahead must leave room for the slice being consumed
    // and for cache turnover while asynchronous requests complete. Real 7B
    // pressure evidence showed that probing into 3/4 or all of a 256 MiB cache
    // with 64 MiB slices can collapse usefulness. Bound adaptive look-ahead to
    // half of the copy cache while honoring any tighter caller budget. Fixed mode
    // keeps the caller's full budget so benchmark sweeps can measure the oracle.
    const std::uint64_t adaptive_cache_limit =
        cfg_.copy_cache ? std::max<std::uint64_t>(1, cfg_.cache_bytes / 2) : configured_budget;
    const std::uint64_t inflight_limit = cfg_.adaptive_prefetch
                                             ? std::min(configured_budget, adaptive_cache_limit)
                                             : configured_budget;

    AdaptivePrefetchController controller(
        {cfg_.adaptive_min_depth,
         cfg_.adaptive_max_depth,
         cfg_.adaptive_low_useful_ratio,
         cfg_.adaptive_high_useful_ratio},
        depth);

    pf.final_depth = depth;
    pf.min_depth_seen = depth;
    pf.max_depth_seen = depth;
    pf.hot_set_budget_bytes = inflight_limit == unlimited ? 0 : inflight_limit;

    std::unordered_map<std::uint32_t, std::uint64_t> requested;
    std::uint64_t inflight_bytes = 0;
    std::uint64_t window_peak_inflight = 0;
    double window_ms = 0.0;
    std::uint32_t window_items = 0;
    std::uint64_t window_consumed = 0;
    std::uint64_t window_useful = 0;
    std::uint64_t window_budget_skips = 0;

    for (std::uint32_t p = 0; p < cfg_.passes; ++p) {
        requested.clear();
        inflight_bytes = 0;

        for (std::uint32_t i = 0; i < store_.count(); ++i) {
            const auto item_start = std::chrono::steady_clock::now();

            if (auto it = requested.find(i); it != requested.end()) {
                const auto bytes = it->second;
                // mmap-only prefetch is advisory and has no copy-cache residency bit to
                // inspect; reaching the requested slice is therefore the useful event.
                const bool ready = !cfg_.copy_cache || cache_.contains(i);
                if (ready) {
                    ++pf.useful;
                    pf.bytes_useful += bytes;
                    ++window_useful;
                } else {
                    ++pf.unused;
                    ++pf.late;
                    pf.bytes_unused += bytes;
                }
                ++window_consumed;
                inflight_bytes -= std::min(inflight_bytes, bytes);
                requested.erase(it);
            }

            for (std::uint32_t d = 1; d <= depth; ++d) {
                if (i + d >= store_.count()) break;
                const auto id = i + d;
                if (requested.find(id) != requested.end()) continue;
                if (cfg_.copy_cache && cache_.contains(id)) continue;

                const auto bytes = store_.slice(id).bytes;
                ++pf.eligible;
                if (inflight_limit != unlimited &&
                    (bytes > inflight_limit || inflight_bytes > inflight_limit - bytes)) {
                    ++pf.skipped_budget;
                    ++window_budget_skips;
                    continue;
                }

                requested.emplace(id, bytes);
                inflight_bytes += bytes;
                window_peak_inflight = std::max(window_peak_inflight, inflight_bytes);
                pf.max_inflight_bytes = std::max(pf.max_inflight_bytes, inflight_bytes);
                ++pf.requests;
                pf.bytes_requested += bytes;
                if (cfg_.copy_cache) {
                    prefetcher_.request(id);
                } else {
                    store_.prefetch(id);
                }
            }

            const auto& s = store_.slice(i);
            const std::byte* ptr = nullptr;
            std::shared_ptr<const std::vector<std::byte>> cache_buf;
            if (cfg_.copy_cache) {
                cache_buf = cache_.get_or_load(i, store_.ptr(i), s.bytes);
                store_.release(i);
                ptr = cache_buf->data();
            } else {
                store_.prefetch(i);
                ptr = store_.ptr(i);
            }

            const auto* u = reinterpret_cast<const unsigned char*>(ptr);
            constexpr std::uint64_t stride = 4096;
            for (std::uint64_t j = 0; j < s.bytes; j += stride) {
                checksum ^= u[j];
                checksum *= 1099511628211ull;
            }
            if (!cfg_.copy_cache) store_.release(i);
            total += s.bytes;

            const auto item_end = std::chrono::steady_clock::now();
            window_ms +=
                std::chrono::duration<double, std::milli>(item_end - item_start).count();
            ++window_items;

            if (cfg_.adaptive_prefetch && window_items >= cfg_.adaptive_window) {
                const double average_item_ms = window_ms / window_items;
                const auto decision = controller.observe({average_item_ms,
                                                          window_consumed,
                                                          window_useful,
                                                          window_budget_skips,
                                                          cache_.stats().evictions,
                                                          window_peak_inflight,
                                                          inflight_limit});
                depth = decision.depth;
                if (decision.adjustment == PrefetchAdjustment::Up) {
                    ++pf.adjustments_up;
                } else if (decision.adjustment == PrefetchAdjustment::Down) {
                    ++pf.adjustments_down;
                }

                window_ms = 0.0;
                window_items = 0;
                window_consumed = 0;
                window_useful = 0;
                window_budget_skips = 0;
                window_peak_inflight = 0;
                pf.min_depth_seen = std::min(pf.min_depth_seen, depth);
                pf.max_depth_seen = std::max(pf.max_depth_seen, depth);
            }
        }

        for (const auto& [id, bytes] : requested) {
            (void)id;
            ++pf.unused;
            pf.bytes_unused += bytes;
        }
        requested.clear();
        inflight_bytes = 0;
    }

    const auto end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(end - start).count();
    prefetcher_.stop();
    prefetcher_.rethrow_if_failed();
    pf.final_depth = depth;
    return {sec, gib(total) / sec, checksum, cache_.stats(), rss_kb(), pf};
}

} // namespace memvanta
