#pragma once
#include "memvanta/tensor_store.hpp"
#include "memvanta/lru_cache.hpp"
#include "memvanta/prefetcher.hpp"
#include <cstdint>
namespace memvanta {
struct RunConfig {
  std::uint64_t cache_bytes;
  std::uint32_t passes=1;
  std::uint32_t prefetch_depth=2;
  bool copy_cache=true;
  bool adaptive_prefetch=false;
  std::uint32_t adaptive_min_depth=1;
  std::uint32_t adaptive_max_depth=4;
  std::uint32_t adaptive_window=8;
};
struct PrefetchStats {
  std::uint64_t requests=0;
  std::uint64_t useful=0;
  std::uint64_t unused=0;
  std::uint64_t adjustments_up=0;
  std::uint64_t adjustments_down=0;
  std::uint32_t final_depth=0;
  std::uint32_t min_depth_seen=0;
  std::uint32_t max_depth_seen=0;
  std::uint64_t hot_set_budget_bytes=0;
};
struct RunStats {
  double seconds=0, gib_per_s=0;
  std::uint64_t checksum=0;
  CacheStats cache;
  std::uint64_t peak_rss_kb=0;
  PrefetchStats prefetch;
};
class Runtime {
public:
  Runtime(const TensorStore& store, RunConfig cfg);
  RunStats run_stream();
private:
  static std::uint64_t rss_kb();
  const TensorStore& store_; RunConfig cfg_; TensorCache cache_; Prefetcher prefetcher_;
};
}
