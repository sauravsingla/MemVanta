#include "memvanta/runtime.hpp"
#include "memvanta/common.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
namespace memvanta {
Runtime::Runtime(const TensorStore&s,RunConfig c):store_(s),cfg_(c),cache_(c.cache_bytes),prefetcher_(s,cache_){
  if(cfg_.adaptive_prefetch){
    if(cfg_.adaptive_min_depth==0) cfg_.adaptive_min_depth=1;
    if(cfg_.adaptive_max_depth<cfg_.adaptive_min_depth) cfg_.adaptive_max_depth=cfg_.adaptive_min_depth;
    if(cfg_.adaptive_window==0) cfg_.adaptive_window=1;
    cfg_.prefetch_depth=std::clamp(cfg_.prefetch_depth,cfg_.adaptive_min_depth,cfg_.adaptive_max_depth);
  }
}
std::uint64_t Runtime::rss_kb(){ std::ifstream f("/proc/self/status"); std::string k; while(f>>k){ if(k=="VmHWM:"){ std::uint64_t v; std::string u; f>>v>>u; return v;} std::string rest; std::getline(f,rest);} return 0; }
RunStats Runtime::run_stream(){
  auto start=std::chrono::steady_clock::now();
  std::uint64_t checksum=1469598103934665603ull,total=0;
  std::uint32_t depth=cfg_.prefetch_depth;
  PrefetchStats pf{};
  pf.final_depth=depth; pf.min_depth_seen=depth; pf.max_depth_seen=depth; pf.hot_set_budget_bytes=cfg_.copy_cache?cfg_.cache_bytes:0;
  std::unordered_set<std::uint32_t> requested;
  double previous_window_ms=std::numeric_limits<double>::infinity(),window_ms=0.0;
  std::uint32_t window_items=0;
  std::uint64_t previous_evictions=0;
  for(std::uint32_t p=0;p<cfg_.passes;++p){
    requested.clear();
    for(std::uint32_t i=0;i<store_.count();++i){
      const auto item_start=std::chrono::steady_clock::now();
      if(requested.erase(i)){
        if(cfg_.copy_cache && cache_.contains(i)) ++pf.useful;
        else ++pf.unused;
      }
      for(std::uint32_t d=1;d<=depth;++d) if(i+d<store_.count()){
        const auto id=i+d;
        if(requested.insert(id).second){
          ++pf.requests;
          if(cfg_.copy_cache) prefetcher_.request(id); else store_.prefetch(id);
        }
      }
      auto&s=store_.slice(i); const std::byte* ptr=nullptr;
      if(cfg_.copy_cache){ auto buf=cache_.get_or_load(i,store_.ptr(i),s.bytes); store_.release(i); ptr=buf->data(); }
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
        const bool eviction_pressure=cfg_.copy_cache && evictions>previous_evictions;
        if(eviction_pressure || (previous_window_ms<std::numeric_limits<double>::infinity() && avg>previous_window_ms*1.05)){
          if(depth>cfg_.adaptive_min_depth){--depth;++pf.adjustments_down;}
        } else if(previous_window_ms<std::numeric_limits<double>::infinity() && avg<=previous_window_ms*1.01 && depth<cfg_.adaptive_max_depth){
          ++depth;++pf.adjustments_up;
        }
        previous_window_ms=avg; previous_evictions=evictions; window_ms=0.0; window_items=0;
        pf.min_depth_seen=std::min(pf.min_depth_seen,depth); pf.max_depth_seen=std::max(pf.max_depth_seen,depth);
      }
    }
    pf.unused+=requested.size();
  }
  auto end=std::chrono::steady_clock::now(); double sec=std::chrono::duration<double>(end-start).count();
  prefetcher_.stop(); pf.final_depth=depth;
  return {sec,gib(total)/sec,checksum,cache_.stats(),rss_kb(),pf};
}
}
