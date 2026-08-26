#include "memvanta/llama_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
double rss_mib(){ rusage r{}; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1024.0; }
std::vector<int> fixed_tokens(std::size_t n,std::size_t vocab){std::vector<int>x(n);for(std::size_t i=0;i<n;++i)x[i]=static_cast<int>((i*1543+17)%std::max<std::size_t>(vocab,1));return x;}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v.size()%2?v[v.size()/2]:(v[v.size()/2-1]+v[v.size()/2])/2.0;}

void drop_model_pages(const memvanta::LlamaModel& m){
    const auto& mf=m.gguf().mapped_file();
    for(const auto& t:m.gguf().tensors()) mf.advise_dontneed(t.offset,t.nbytes);
}

void prefetch_layers(const memvanta::LlamaModel& m,std::size_t depth){
    if(!depth)return;
    const auto& mf=m.gguf().mapped_file();
    const auto& ts=m.gguf().tensors();
    std::size_t layers=m.config().n_layer;
    depth=std::min(depth,layers);
    for(const auto& t:ts){
        if(t.name=="token_embd.weight"||t.name=="output.weight"||t.name=="output_norm.weight") mf.advise_willneed(t.offset,t.nbytes);
    }
    for(std::size_t li=0;li<depth;++li){
        const std::string p="blk."+std::to_string(li)+".";
        for(const auto& t:ts) if(t.name.rfind(p,0)==0) mf.advise_willneed(t.offset,t.nbytes);
    }
}
}

int main(int argc,char**argv){
    try{
        std::string model,mode="none",csv; unsigned threads=4,reps=5; std::size_t ctx=128,prompt_n=32,gen_n=8,depth=2,min_depth=1,max_depth=4;
        for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
            if(a=="--model")model=val();else if(a=="--mode")mode=val();else if(a=="--threads")threads=std::stoul(val());else if(a=="--reps")reps=std::stoul(val());else if(a=="--ctx")ctx=std::stoull(val());else if(a=="--prompt")prompt_n=std::stoull(val());else if(a=="--gen")gen_n=std::stoull(val());else if(a=="--depth")depth=std::stoull(val());else if(a=="--min-depth")min_depth=std::stoull(val());else if(a=="--max-depth")max_depth=std::stoull(val());else if(a=="--csv")csv=val();else throw std::runtime_error("unknown arg: "+a);}
        if(model.empty())throw std::runtime_error("--model required"); if(mode!="none"&&mode!="fixed"&&mode!="adaptive")throw std::runtime_error("--mode must be none|fixed|adaptive");
        memvanta::LlamaModel m(model,threads,ctx); prompt_n=std::min(prompt_n,m.config().n_ctx/2);gen_n=std::min(gen_n,m.config().n_ctx/4);auto prompt=fixed_tokens(prompt_n,m.config().vocab);auto seed=fixed_tokens(std::min<std::size_t>(16,m.config().n_ctx-gen_n),m.config().vocab);memvanta::Sampler greedy({0,1,1});
        std::vector<double>pp,tg;std::size_t current=std::clamp(depth,min_depth,max_depth);double prev_total=0;
        for(unsigned r=0;r<reps;++r){
            drop_model_pages(m); if(mode=="fixed")prefetch_layers(m,depth);else if(mode=="adaptive")prefetch_layers(m,current);
            m.reset();auto a=Clock::now();m.prefill(prompt,16,true);auto b=Clock::now();double pp_s=std::chrono::duration<double>(b-a).count();pp.push_back(prompt.size()/pp_s);
            m.reset();m.prefill(seed,16,false);int cur=17%m.config().vocab;auto c=Clock::now();for(std::size_t i=0;i<gen_n;++i)cur=greedy.sample(m.forward(cur,true));auto d=Clock::now();double tg_s=std::chrono::duration<double>(d-c).count();tg.push_back(gen_n/tg_s);
            double total=pp_s+tg_s;if(mode=="adaptive"&&r>0){if(total>prev_total*1.03&&current>min_depth)--current;else if(total<prev_total*0.97&&current<max_depth)++current;}prev_total=total;
        }
        std::cout<<"mode="<<mode<<" threads="<<threads<<" pp_tps_median="<<median(pp)<<" tg_tps_median="<<median(tg)<<" peak_rss_mib="<<rss_mib()<<" final_depth="<<current<<"\n";
        if(!csv.empty()){std::ofstream f(csv);f<<"rep,pp_tps,tg_tps\n";for(std::size_t i=0;i<pp.size();++i)f<<i+1<<","<<pp[i]<<","<<tg[i]<<"\n";}
        return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
}
