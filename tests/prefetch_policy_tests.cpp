#include "check.hpp"
#include "memvanta/prefetch_policy.hpp"

#include <cstdint>
#include <limits>

using memvanta::AdaptivePrefetchController;
using memvanta::AdaptivePrefetchPolicyConfig;
using memvanta::AdaptivePrefetchWindow;
using memvanta::PrefetchAdjustment;

namespace {

AdaptivePrefetchWindow good_window(double ms,
                                   std::uint64_t evictions = 0,
                                   bool has_late_request = true) {
    AdaptivePrefetchWindow w;
    w.average_item_ms = ms;
    if (has_late_request) {
        // Keep usefulness at the high threshold while providing a concrete
        // signal that one more unit of look-ahead may help.
        w.consumed = 10;
        w.useful = 9;
        w.late = 1;
    } else {
        w.consumed = 8;
        w.useful = 8;
        w.late = 0;
    }
    w.evictions = evictions;
    w.peak_inflight_bytes = 32;
    w.inflight_limit = 256;
    return w;
}

} // namespace

int main() {
    {
        AdaptivePrefetchController c({1, 4, 0.60, 0.90}, 2);
        auto w = good_window(10.0);
        w.budget_skips = 1;
        const auto d = c.observe(w);
        CHECK(d.depth == 1);
        CHECK(d.adjustment == PrefetchAdjustment::Down);
    }

    {
        // Fully timely prefetch is already satisfying demand. It must not spend
        // RSS or latency on speculative deeper probes merely because timing is
        // stable and usefulness is 100%.
        AdaptivePrefetchController c({1, 3, 0.60, 0.90}, 1);
        const auto timely = good_window(10.0, 0, false);
        for (int i = 0; i < 12; ++i) {
            const auto d = c.observe(timely);
            CHECK(d.depth == 1);
            CHECK(d.adjustment == PrefetchAdjustment::None);
        }
    }

    {
        AdaptivePrefetchController c({1, 3, 0.60, 0.90}, 1);
        CHECK(c.observe(good_window(10.0)).depth == 1);
        CHECK(c.observe(good_window(10.0)).depth == 1);
        const auto probe = c.observe(good_window(10.0));
        CHECK(probe.depth == 2);
        CHECK(probe.adjustment == PrefetchAdjustment::Up);

        const auto rollback = c.observe(good_window(10.10));
        CHECK(rollback.depth == 1);
        CHECK(rollback.adjustment == PrefetchAdjustment::Down);

        // A rejected 1->2 probe remains blocked throughout the same steady
        // phase instead of repeatedly paying the memory/latency cost.
        for (int i = 0; i < 20; ++i) {
            const auto d = c.observe(good_window(10.0));
            CHECK(d.depth == 1);
            CHECK(d.adjustment == PrefetchAdjustment::None);
        }
    }

    {
        AdaptivePrefetchController c({1, 3, 0.60, 0.90}, 1);
        c.observe(good_window(10.0));
        c.observe(good_window(10.0));
        CHECK(c.observe(good_window(10.0)).depth == 2);
        CHECK(c.observe(good_window(10.10)).depth == 1);

        // A material slowdown at the proven lower depth is a phase change. The
        // old rejection is then allowed to expire and one fresh probe may run.
        c.observe(good_window(10.60));
        bool reprobed = false;
        for (int i = 0; i < 8; ++i) {
            const auto d = c.observe(good_window(10.60));
            if (d.adjustment == PrefetchAdjustment::Up) {
                reprobed = true;
                break;
            }
        }
        CHECK(reprobed);
    }

    {
        AdaptivePrefetchController c({1, 3, 0.60, 0.90}, 1);
        c.observe(good_window(10.0));
        c.observe(good_window(10.0));
        const auto probe = c.observe(good_window(10.0));
        CHECK(probe.depth == 2);

        // A highly useful probe within 0.5% of the reference is retained. The
        // byte ceiling, not timing noise, remains the memory guardrail.
        const auto keep = c.observe(good_window(10.04));
        CHECK(keep.depth == 2);
        CHECK(keep.adjustment == PrefetchAdjustment::None);
    }

    {
        AdaptivePrefetchController c({1, 4, 0.60, 0.90}, 3);
        auto w = good_window(10.0, 1);
        w.useful = 1;
        const auto d = c.observe(w);
        CHECK(d.depth == 2);
        CHECK(d.adjustment == PrefetchAdjustment::Down);
    }

    {
        AdaptivePrefetchController c({1, 4, 0.60, 0.90}, 2);
        auto w = good_window(10.0);
        w.peak_inflight_bytes = 240;
        c.observe(w);
        c.observe(w);
        const auto d = c.observe(w);
        CHECK(d.depth == 2);
        CHECK(d.adjustment == PrefetchAdjustment::None);
    }

    return memvanta_test::failures();
}
