#include "memvanta/common.hpp"
#include "memvanta/lru_cache.hpp"
#include "memvanta/quant.hpp"
#include "memvanta/llama_model.hpp"
#include "memvanta/runtime.hpp"
#include "memvanta/tensor_store.hpp"
#include "memvanta/worker_pool.hpp"
#include "check.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
int main(){
  CHECK(memvanta::parse_size("1M")==1048576);
  {
    std::vector<std::byte> src(64); memvanta::TensorCache c(128);
    auto a=c.get_or_load(1,src.data(),64); auto b=c.get_or_load(1,src.data(),64);
    auto st=c.stats(); CHECK(st.hits==1 && st.misses==1); (void)a;(void)b;
  }
  {
    bool threw=false;try{(void)memvanta::ggml_tensor_nbytes(memvanta::GgmlType::F32,std::numeric_limits<std::uint64_t>::max());}catch(const std::runtime_error&){threw=true;}
    CHECK_MSG(threw,"GGUF byte-size overflow was not rejected");
    memvanta::GgufTensor t;t.dims={64,0};threw=false;try{(void)t.elements();}catch(const std::runtime_error&){threw=true;}
    CHECK_MSG(threw,"zero tensor dimension was not rejected");
  }
  {
    constexpr std::size_t n=256; std::vector<float>a(n),x(n);
    std::mt19937 g(7); std::uniform_real_distribution<float>d(-1,1);
    float ref=0; for(std::size_t i=0;i<n;++i){a[i]=d(g);x[i]=d(g);ref+=a[i]*x[i];}
    auto q8=memvanta::quantize_q8_0(a.data(),n); auto q4=memvanta::quantize_q4_0(a.data(),n);
    float e8=std::fabs(memvanta::dot_q8_0(q8.data(),x.data(),n)-ref)/(std::fabs(ref)+1e-6f);
    float e4=std::fabs(memvanta::dot_q4_0(q4.data(),x.data(),n)-ref)/(std::fabs(ref)+1e-6f);
    CHECK_MSG(e8 < 0.08f,"Q8_0 dot product exceeded its quantization error budget");
    CHECK_MSG(e4 < 0.35f,"Q4_0 dot product exceeded its quantization error budget");
  }
  {
    memvanta::WorkerPool pool(4); std::vector<int> seen(100,0);
    pool.parallel_for(seen.size(),[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)seen[i]=1;});
    // Report one failure with a count rather than one per unvisited index.
    const auto missed=static_cast<std::size_t>(std::count(seen.begin(),seen.end(),0));
    CHECK_MSG(missed==0,"WorkerPool::parallel_for left indices unvisited");
    bool threw=false;try{pool.parallel_for(64,[&](std::size_t a,std::size_t){if(a>0)throw std::runtime_error("worker failure");});}catch(const std::runtime_error&){threw=true;}
    CHECK_MSG(threw,"WorkerPool did not propagate a worker exception");
    std::fill(seen.begin(),seen.end(),0);pool.parallel_for(seen.size(),[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)seen[i]=1;});
    CHECK_MSG(std::count(seen.begin(),seen.end(),0)==0,"WorkerPool did not recover after a propagated exception");
  }
  {
    constexpr std::size_t d=64; std::vector<float> k(d),v(d),q(d),out(d),refv(d); std::mt19937 g(11); std::uniform_real_distribution<float> dist(-1,1);
    for(std::size_t i=0;i<d;++i){k[i]=dist(g);v[i]=dist(g);q[i]=dist(g);refv[i]=0.25f*v[i];}
    float refdot=0;for(std::size_t i=0;i<d;++i)refdot+=k[i]*q[i];
    for(auto typ:{memvanta::KVCacheType::F32,memvanta::KVCacheType::F16,memvanta::KVCacheType::Q8}){
      memvanta::PagedKVCache c(d,16,typ);c.write(0,k.data(),v.data());std::fill(out.begin(),out.end(),0);float got=c.dot_key(0,0,q.data(),d);c.add_value(0,0,0.25f,out.data(),d);
      float de=std::fabs(got-refdot)/(std::fabs(refdot)+1e-5f),ve=0,base=0;for(std::size_t i=0;i<d;++i){ve+=std::fabs(out[i]-refv[i]);base+=std::fabs(refv[i]);}ve/=base+1e-5f;
      const char* who=memvanta::kv_cache_type_name(typ);
      if(typ==memvanta::KVCacheType::F32){CHECK_MSG(de<1e-5f,who);CHECK_MSG(ve<1e-5f,who);}
      else if(typ==memvanta::KVCacheType::F16){CHECK_MSG(de<0.01f,who);CHECK_MSG(ve<0.01f,who);}
      else{CHECK_MSG(de<0.08f,who);CHECK_MSG(ve<0.03f,who);}
    }
  }
  {
    const char* path="memvanta_adaptive_prefetch_test.bin";
    { std::ofstream f(path,std::ios::binary); for(int i=0;i<32768;++i){ unsigned char b=static_cast<unsigned char>(i*17); f.write(reinterpret_cast<const char*>(&b),1); } }
    memvanta::TensorStore store(path,4096);
    memvanta::RunConfig fixed{16384,2,2,true};
    auto a=memvanta::Runtime(store,fixed).run_stream();
    memvanta::RunConfig adaptive{16384,2,2,true}; adaptive.adaptive_prefetch=true; adaptive.adaptive_min_depth=1; adaptive.adaptive_max_depth=3; adaptive.adaptive_window=2;
    auto b=memvanta::Runtime(store,adaptive).run_stream();
    CHECK_MSG(a.checksum==b.checksum,"adaptive prefetch changed the streamed result");
    CHECK(b.prefetch.final_depth>=1 && b.prefetch.final_depth<=3);
    CHECK(b.prefetch.min_depth_seen>=1 && b.prefetch.max_depth_seen<=3);
    CHECK(b.prefetch.hot_set_budget_bytes==16384);
    CHECK(b.prefetch.useful+b.prefetch.unused<=b.prefetch.requests);
    std::remove(path);
  }
  if(memvanta_test::failures()){ std::cerr<<"FAILED\n"; return 1; }
  std::cout<<"ok\n";
  return 0;
}
