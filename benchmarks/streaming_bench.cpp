#include "memvanta/runtime.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::uint64_t mib(const std::string& s) {
    return std::stoull(s) << 20;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: memvanta_bench FILE [--mode none|fixed|adaptive] [--slice-mib N] "
                         "[--cache-mib N] [--passes N] [--depth N] [--min-depth N] [--max-depth N] "
                         "[--window N] [--budget-mib N]\n";
            return 1;
        }
        std::string path = argv[1], mode = "adaptive";
        std::uint64_t slice_bytes = 64ull << 20, cache_bytes = 256ull << 20,
                      budget_bytes = 256ull << 20;
        std::uint32_t passes = 1, depth = 2, min_depth = 1, max_depth = 4, window = 8;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--mode")
                mode = value();
            else if (a == "--slice-mib")
                slice_bytes = mib(value());
            else if (a == "--cache-mib")
                cache_bytes = mib(value());
            else if (a == "--budget-mib")
                budget_bytes = mib(value());
            else if (a == "--passes")
                passes = static_cast<std::uint32_t>(std::stoul(value()));
            else if (a == "--depth")
                depth = static_cast<std::uint32_t>(std::stoul(value()));
            else if (a == "--min-depth")
                min_depth = static_cast<std::uint32_t>(std::stoul(value()));
            else if (a == "--max-depth")
                max_depth = static_cast<std::uint32_t>(std::stoul(value()));
            else if (a == "--window")
                window = static_cast<std::uint32_t>(std::stoul(value()));
            else
                throw std::runtime_error("unknown argument: " + a);
        }
        if (mode != "none" && mode != "fixed" && mode != "adaptive")
            throw std::runtime_error("mode must be none, fixed, or adaptive");
        if (!slice_bytes || !cache_bytes || !passes)
            throw std::runtime_error("slice/cache/passes must be non-zero");

        memvanta::TensorStore store(path, slice_bytes);
        memvanta::RunConfig cfg{};
        cfg.cache_bytes = cache_bytes;
        cfg.passes = passes;
        cfg.prefetch_depth = mode == "none" ? 0 : depth;
        cfg.copy_cache = true;
        cfg.adaptive_prefetch = mode == "adaptive";
        cfg.adaptive_min_depth = min_depth;
        cfg.adaptive_max_depth = max_depth;
        cfg.adaptive_window = window;
        cfg.prefetch_budget_bytes = budget_bytes;
        auto s = memvanta::Runtime(store, cfg).run_stream();
        const auto feedback = s.prefetch.useful + s.prefetch.unused;
        const double useful_ratio = feedback ? double(s.prefetch.useful) / double(feedback) : 0.0;
        std::cout << std::fixed << std::setprecision(6) << "mode=" << mode
                  << " throughput_gib_s=" << s.gib_per_s << " seconds=" << s.seconds
                  << " peak_rss_mib=" << (s.peak_rss_kb / 1024.0) << " checksum=" << s.checksum
                  << " eligible=" << s.prefetch.eligible << " requests=" << s.prefetch.requests
                  << " useful=" << s.prefetch.useful << " unused=" << s.prefetch.unused
                  << " late=" << s.prefetch.late << " useful_ratio=" << useful_ratio
                  << " bytes_requested=" << s.prefetch.bytes_requested
                  << " bytes_useful=" << s.prefetch.bytes_useful
                  << " bytes_unused=" << s.prefetch.bytes_unused
                  << " skipped_budget=" << s.prefetch.skipped_budget
                  << " peak_inflight_bytes=" << s.prefetch.max_inflight_bytes
                  << " budget_bytes=" << s.prefetch.hot_set_budget_bytes
                  << " final_depth=" << s.prefetch.final_depth
                  << " min_depth=" << s.prefetch.min_depth_seen
                  << " max_depth=" << s.prefetch.max_depth_seen
                  << " adjustments_up=" << s.prefetch.adjustments_up
                  << " adjustments_down=" << s.prefetch.adjustments_down
                  << " cache_hits=" << s.cache.hits << " cache_misses=" << s.cache.misses
                  << " cache_evictions=" << s.cache.evictions << "\n";
        if (s.prefetch.hot_set_budget_bytes &&
            s.prefetch.max_inflight_bytes > s.prefetch.hot_set_budget_bytes)
            throw std::runtime_error("prefetch byte ceiling was exceeded");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
