#include "memvanta/runtime.hpp"
#include "memvanta/common.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <sys/resource.h>
#include <unordered_map>
namespace memvanta {
Runtime::Runtime(const TensorStore&s,RunConfig c):store_(s),cfg_(c),cache_(c.cache_bytes),prefetcher_(s,cache_){
  if(cfg_.copy_cache && cfg_.prefetch_budget_bytes>cfg_.cache_bytes) cfg_.prefetch_budget_bytes=cfg_.cache_bytes;
  if(cfg_.adaptive_prefetch){
    if(cfg_.adaptive_min_depth==0) cfg_.adaptive_min_depth=1;
    if(cfg_.adaptive_max_depth<cfg_.adaptive_min_depth) cfg_.adaptive_max_depth=cfg_.adaptive_min_depth;
    if(cfg_.adaptive_window==0) cfg_.adaptive_window=1;
    cfg_.prefetch_depth=std::clamp(cfg_.prefetch_depth,cfg_.adaptive_min_depth,cfg_.adaptive_max_depth);
  }
}
std::uint64_t Runtime::rss_kb(){
  rusage r{};
  if(getrusage(RUSAGE_SELF,&r)!=0) return 0;
#if defined(__APPLE__)
  return static_cast<std::uint64_t>(r.ru_maxrss/1024);
#else
  return static_cast<std::uint64_t>(r.ru_maxrss);
#endif
}
RunStats Runtime::run_stream(){
  auto start=std::chrono::steady_clock::now();
  std::uint64_t checksum=1469598103934665603ull,total=0;
  std::uint32_t depth=cfg_.prefetch_depth;
  PrefetchStats pf{};
  const std::uint64_t unlimited=std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t budget=cfg_.prefetch_budget_bytes?cfg_.prefetch_budget_bytes:(cfg_.copy_cache?cfg_.cache_bytes:unlimited);
  pf.final_depth=depth; pf.min_depth_seen=depth; pf.max_depth_seen=depth;
  pf.hot_set_budget_bytes=budget==unlimited?0:budget;
  std::unordered_map<std::uint32_t,std::uint64_t> requested;
  std::uint64_t inflight_bytes=0;
  double previous_window_ms=std::numeric_limits<double>::infinity(),window_ms=0.0;
  std::uint32_t window_items=0;
  std::uint64_t window_consumed=0,window_useful=0,previous_evictions=0;
  for(std::uint32_t p=0;p<cfg_.passes;++p){
    requested.clear(); inflight_bytes=0;
    for(std::uint32_t i=0;i<store_.count();++i){
      const auto item_start=std::chrono::steady_clock::now();
      if(auto it=requested.find(i);it!=requested.end()){
        const auto bytes=it->second;
        const bool ready=cfg_.copy_cache&&cache_.contains(i);
        if(ready){++pf.useful;pf.bytes_useful+=bytes;++window_useful;}
        else {++pf.unused;++pf.late;pf.bytes_unused+=bytes;}
        ++window_consumed;
        inflight_bytes-=std::min(inflight_bytes,bytes);
        requested.erase(it);
      }
      for(std::uint32_t d=1;d<=depth;++d) if(i+d<store_.count()){
        const auto id=i+d;
        if(requested.find(id)!=requested.end())continue;
        if(cfg_.copy_cache&&cache_.contains(id))continue;
        const auto bytes=store_.slice(id).bytes;
        ++pf.eligible;
        if(budget!=unlimited&&(bytes>budget||inflight_bytes>budget-bytes)){
          ++pf.skipped_budget;
          continue;
        }
        requested.emplace(id,bytes);
        inflight_bytes+=bytes;
        pf.max_inflight_bytes=std::max(pf.max_inflight_bytes,inflight_bytes);
        ++pf.requests;pf.bytes_requested+=bytes;
        if(cfg_.copy_cache) prefetcher_.request(id); else store_.prefetch(id);
      }
      auto&s=store_.slice(i); const std::byte* ptr=nullptr;
      std::shared_ptr<const std::vector<std::byte>> cache_buf;
      if(cfg_.copy_cache){
        cache_buf=cache_.get_or_load(i,store_.ptr(i),s.bytes);
        store_.release(i);
        ptr=cache_buf->data();
      }
      else { store_.prefetch(i); ptr=store_.ptr(i); }
      const auto* u=reinterpret_cast<const unsigned char*>(ptr); std::uint64_t stride=4096;
      for(std::uint64_t j=0;j<s.bytes;j+=stride){ checksum^=u[j]; checksum*=1099511628211ull; }
      if(!cfg_.copy_cache) store_.release(i);
      total+=s.bytes;

      const auto item_end=std::chrono::steady_clock::now();
      window_ms+=std::chrono::duration<double,std::milli>(item_end-item_start).count();
      ++window_items;
      if(cfg_.adaptive_prefetch && window_items>=cfg_.adaptive_window){
        const double avg=window_ms/window_items;
        const auto evictions=cache_.stats().evictions;
        const bool eviction_pressure=cfg_.copy_cache&&evictions>previous_evictions;
        const bool have_usefulness=window_consumed>=2;
        const double useful_ratio=window_consumed?double(window_useful)/double(window_consumed):1.0;
        const bool low_usefulness=have_usefulness&&useful_ratio<0.50;
        const bool high_usefulness=have_usefulness&&useful_ratio>=0.75;
        const bool latency_regression=previous_window_ms<std::numeric_limits<double>::infinity()&&avg>previous_window_ms*1.05;
        const bool latency_stable=previous_window_ms<std::numeric_limits<double>::infinity()&&avg<=previous_window_ms*1.01;
        if(eviction_pressure||low_usefulness||latency_regression){
          if(depth>cfg_.adaptive_min_depth){--depth;++pf.adjustments_down;}
        } else if(high_usefulness&&latency_stable&&depth<cfg_.adaptive_max_depth&&(budget==unlimited||inflight_bytes<budget)){
          ++depth;++pf.adjustments_up;
        }
        previous_window_ms=avg; previous_evictions=evictions; window_ms=0.0; window_items=0; window_consumed=0; window_useful=0;
        pf.min_depth_seen=std::min(pf.min_depth_seen,depth); pf.max_depth_seen=std::max(pf.max_depth_seen,depth);
      }
    }
    for(const auto& [id,bytes]:requested){(void)id;++pf.unused;pf.bytes_unused+=bytes;}
    requested.clear();inflight_bytes=0;
  }
  auto end=std::chrono::steady_clock::now(); double sec=std::chrono::duration<double>(end-start).count();
  prefetcher_.stop();
  prefetcher_.rethrow_if_failed();
  pf.final_depth=depth;
  return {sec,gib(total)/sec,checksum,cache_.stats(),rss_kb(),pf};
}
}