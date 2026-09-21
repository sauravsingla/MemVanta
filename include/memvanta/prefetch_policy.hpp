#pragma once

#include <cstdint>
#include <limits>

namespace memvanta {

enum class PrefetchAdjustment { None, Up, Down };

struct AdaptivePrefetchPolicyConfig {
    std::uint32_t min_depth{1};
    std::uint32_t max_depth{4};
    double low_useful_ratio{0.60};
    double high_useful_ratio{0.90};
};

struct AdaptivePrefetchWindow {
    double average_item_ms{};
    std::uint64_t consumed{};
    std::uint64_t useful{};
    std::uint64_t budget_skips{};
    std::uint64_t evictions{};
    std::uint64_t peak_inflight_bytes{};
    std::uint64_t inflight_limit{std::numeric_limits<std::uint64_t>::max()};
};

struct AdaptivePrefetchDecision {
    std::uint32_t depth{};
    PrefetchAdjustment adjustment{PrefetchAdjustment::None};
};

class AdaptivePrefetchController {
public:
    AdaptivePrefetchController(AdaptivePrefetchPolicyConfig config, std::uint32_t initial_depth);

    AdaptivePrefetchDecision observe(const AdaptivePrefetchWindow& window);
    std::uint32_t depth() const { return depth_; }

private:
    AdaptivePrefetchPolicyConfig config_;
    std::uint32_t depth_{};
    std::uint32_t stable_windows_{};
    std::uint32_t cooldown_windows_{};
    std::uint32_t probe_from_depth_{};
    std::uint64_t previous_evictions_{};
    double previous_window_ms_{std::numeric_limits<double>::infinity()};
    double probe_reference_ms_{};
    bool probing_up_{};
};

} // namespace memvanta
