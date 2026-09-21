#include "memvanta/runtime.hpp"
#include "memvanta/tensor_store.hpp"
#include "memvanta/worker_pool.hpp"
#include "check.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void phase(const char* name){std::cout<<"[phase] "<<name<<'\n'<<std::flush;}
}

int main(){
    phase("worker-pool-reuse");
    {
        memvanta::WorkerPool pool(4);
        std::vector<int> seen(257);
        for(int iter=0;iter<1000;++iter){
            std::fill(seen.begin(),seen.end(),0);
            pool.parallel_for(seen.size(),[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)seen[i]=iter+1;});
            CHECK_MSG(std::count(seen.begin(),seen.end(),iter+1)==static_cast<std::ptrdiff_t>(seen.size()),"WorkerPool stress run left indices unvisited");
        }
    }

    phase("worker-pool-exception-recovery");
    {
        memvanta::WorkerPool pool(4);
        std::vector<int> seen(128);
        for(int iter=0;iter<250;++iter){
            bool threw=false;
            try{
                pool.parallel_for(128,[&](std::size_t a,std::size_t){if(a>0)throw std::runtime_error("intentional worker failure");});
            }catch(const std::runtime_error&){threw=true;}
            CHECK_MSG(threw,"WorkerPool stress exception was not propagated");
            std::fill(seen.begin(),seen.end(),0);
            pool.parallel_for(seen.size(),[&](std::size_t a,std::size_t b){for(std::size_t i=a;i<b;++i)seen[i]=1;});
            CHECK_MSG(std::count(seen.begin(),seen.end(),1)==static_cast<std::ptrdiff_t>(seen.size()),"WorkerPool did not recover after stress exception");
        }
    }

    phase("runtime-prefetch-lifecycle");
    const char* path="memvanta_concurrency_stress.bin";
    {
        std::ofstream f(path,std::ios::binary);
        for(int i=0;i<65536;++i){const unsigned char b=static_cast<unsigned char>((i*37)^0x5a);f.write(reinterpret_cast<const char*>(&b),1);}
    }
    std::uint64_t reference=0;
    {
        // Keep the mapping stable here so this target stresses Runtime/Prefetcher
        // construction, stop/join, and adaptive-depth changes rather than mixing
        // that signal with repeated mmap/madvise/open/close kernel churn.
        memvanta::TensorStore store(path,4096);
        for(int iter=0;iter<64;++iter){
            if((iter%8)==0)std::cout<<"[runtime-iter] "<<iter<<'\n'<<std::flush;
            memvanta::RunConfig cfg{16384,1,2,true};
            cfg.adaptive_prefetch=(iter%2)==0;
            cfg.adaptive_min_depth=1;
            cfg.adaptive_max_depth=4;
            cfg.adaptive_window=2;
            memvanta::Runtime runtime(store,cfg);
            auto stats=runtime.run_stream();
            if(iter==0)reference=stats.checksum;
            CHECK_MSG(stats.checksum==reference,"runtime/prefetch stress changed deterministic checksum");
        }
    }
    std::remove(path);

    phase("done");
    if(memvanta_test::failures()){std::cerr<<"FAILED\n";return 1;}
    std::cout<<"concurrency stress tests ok\n";
    return 0;
}
