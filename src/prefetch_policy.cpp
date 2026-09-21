#include "memvanta/prefetch_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace memvanta {

AdaptivePrefetchController::AdaptivePrefetchController(AdaptivePrefetchPolicyConfig config,
                                                       std::uint32_t initial_depth)
    : config_(config) {
    config_.low_useful_ratio = std::clamp(config_.low_useful_ratio, 0.0, 1.0);
    config_.high_useful_ratio =
        std::clamp(config_.high_useful_ratio, config_.low_useful_ratio, 1.0);
    if (config_.min_depth == 0) config_.min_depth = 1;
    if (config_.max_depth < config_.min_depth) config_.max_depth = config_.min_depth;
    depth_ = std::clamp(initial_depth, config_.min_depth, config_.max_depth);
    probe_from_depth_ = depth_;
}

AdaptivePrefetchDecision AdaptivePrefetchController::observe(const AdaptivePrefetchWindow& window) {
    if (!std::isfinite(window.average_item_ms) || window.average_item_ms < 0.0) {
        throw std::runtime_error("adaptive prefetch window latency must be finite and non-negative");
    }

    const bool have_usefulness = window.consumed >= 2;
    const double useful_ratio =
        window.consumed ? double(window.useful) / double(window.consumed) : 1.0;
    const bool low_usefulness =
        have_usefulness && useful_ratio < config_.low_useful_ratio;
    const bool high_usefulness =
        have_usefulness && useful_ratio >= config_.high_useful_ratio;

    // Sequential streaming naturally evicts old cache entries. Only treat
    // turnover as pressure when the look-ahead itself is not proving useful.
    const bool eviction_pressure =
        window.evictions > previous_evictions_ && have_usefulness && !high_usefulness;
    const bool slower = std::isfinite(previous_window_ms_) &&
                        window.average_item_ms > previous_window_ms_ * 1.05;
    const bool stable = std::isfinite(previous_window_ms_) &&
                        window.average_item_ms <= previous_window_ms_ * 1.01;
    const bool hard_pressure = eviction_pressure || window.budget_skips || low_usefulness;

    PrefetchAdjustment adjustment = PrefetchAdjustment::None;

    if (probing_up_) {
        // A larger depth is a bounded experiment. Keep it when it is effectively
        // non-regressive (within 0.5%) and usefulness/memory signals remain
        // healthy. If the probe is rejected, remember that exact transition so
        // a homogeneous stream does not repeatedly pay for the same failed
        // experiment. A material (>5%) slowdown at the proven lower depth is
        // treated as a phase change and permits one fresh probe later.
        const bool probe_non_regressive = window.average_item_ms <= probe_reference_ms_ * 1.005;
        if (hard_pressure || !probe_non_regressive) {
            if (depth_ > probe_from_depth_) {
                depth_ = probe_from_depth_;
                adjustment = PrefetchAdjustment::Down;
            }
            previous_window_ms_ = probe_reference_ms_;
            rejected_probe_from_depth_ = probe_from_depth_;
            rejected_probe_reference_ms_ = probe_reference_ms_;
            rejected_probe_blocked_ = true;
            cooldown_windows_ = 2;
        } else {
            previous_window_ms_ = window.average_item_ms;
            cooldown_windows_ = 2;
        }
        probing_up_ = false;
        stable_windows_ = 0;
    } else if (hard_pressure || slower) {
        if (depth_ > config_.min_depth) {
            --depth_;
            adjustment = PrefetchAdjustment::Down;
        }
        previous_window_ms_ = window.average_item_ms;
        stable_windows_ = 0;
        cooldown_windows_ = 2;
    } else {
        previous_window_ms_ = window.average_item_ms;
        if (cooldown_windows_) --cooldown_windows_;
        if (stable && high_usefulness) {
            ++stable_windows_;
        } else {
            stable_windows_ = 0;
        }

        if (rejected_probe_blocked_ && depth_ == rejected_probe_from_depth_ &&
            std::isfinite(rejected_probe_reference_ms_) &&
            window.average_item_ms > rejected_probe_reference_ms_ * 1.05) {
            // The lower-depth baseline has materially changed, so the previous
            // rejection is stale. Re-open the transition after a short cooldown.
            rejected_probe_blocked_ = false;
            cooldown_windows_ = std::max<std::uint32_t>(cooldown_windows_, 2);
            stable_windows_ = 0;
        }

        // Require two stable windows and 25% headroom inside the cache-safe
        // limit before probing one additional depth. A transition that already
        // failed in the current steady phase remains blocked until the phase-
        // change condition above is observed.
        const auto unlimited = std::numeric_limits<std::uint64_t>::max();
        const bool byte_headroom =
            window.inflight_limit == unlimited ||
            (window.inflight_limit > 0 &&
             window.peak_inflight_bytes <
                 window.inflight_limit - (window.inflight_limit / 4));
        const bool rejected_transition =
            rejected_probe_blocked_ && depth_ == rejected_probe_from_depth_;
        if (stable_windows_ >= 2 && !cooldown_windows_ && high_usefulness && byte_headroom &&
            !rejected_transition && depth_ < config_.max_depth) {
            probe_reference_ms_ = window.average_item_ms;
            probe_from_depth_ = depth_;
            ++depth_;
            adjustment = PrefetchAdjustment::Up;
            probing_up_ = true;
            stable_windows_ = 0;
        }
    }

    previous_evictions_ = window.evictions;
    return {depth_, adjustment};
}

} // namespace memvanta
