#include "memvanta/llama_model.hpp"
#include "memvanta/gguf_kernels.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h>
#endif

using Clock = std::chrono::steady_clock;
static std::uint64_t read_cycles(){
#if defined(__i386__) || defined(__x86_64__)
    unsigned aux=0; return __rdtscp(&aux);
#else
    return 0;
#endif
}

int main(int argc,char**argv){
    try{
        std::string model,csv="memvanta-layer-profile.csv";unsigned threads=4;std::size_t ctx=128,prompt_n=32,gen_n=8,batch=16;
        for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};
            if(a=="--model")model=val();else if(a=="--threads")threads=std::stoul(val());else if(a=="--ctx")ctx=std::stoull(val());else if(a=="--prompt")prompt_n=std::stoull(val());else if(a=="--gen")gen_n=std::stoull(val());else if(a=="--batch")batch=std::stoull(val());else if(a=="--csv")csv=val();else throw std::runtime_error("unknown arg: "+a);
        }
        if(model.empty())throw std::runtime_error("--model required");
        memvanta::LlamaModel m(model,threads,ctx,128,memvanta::KVCacheType::F16);
        std::string seed="Once upon a time there was a small village with a curious child who loved stories. ";
        std::vector<int> ids;while(ids.size()<prompt_n){auto z=m.tokenizer().encode(seed,ids.empty());ids.insert(ids.end(),z.begin(),z.end());}ids.resize(std::min(prompt_n,m.config().n_ctx>gen_n?m.config().n_ctx-gen_n:prompt_n));
        memvanta::Sampler greedy({0.0f,1,42});
        memvanta::enable_kernel_profiling(true);memvanta::reset_kernel_profile();m.reset();
        const auto c0=read_cycles();auto t0=Clock::now();m.prefill(ids,batch,true);auto t1=Clock::now();const auto c1=read_cycles();
        auto prefill_entries=memvanta::kernel_profile_snapshot();
        memvanta::reset_kernel_profile();
        int cur=ids.empty()?0:ids.back();double sampling_ms=0.0;
        for(std::size_t i=0;i<gen_n;++i){const auto&logits=m.forward(cur,true);auto s0=Clock::now();cur=greedy.sample(logits);auto s1=Clock::now();sampling_ms+=std::chrono::duration<double,std::milli>(s1-s0).count();}
        const auto c2=read_cycles();auto t2=Clock::now();auto decode_entries=memvanta::kernel_profile_snapshot();memvanta::enable_kernel_profiling(false);

        using Agg=std::map<std::pair<int,memvanta::KernelProfileKind>,double>;
        Agg agg,prefill_agg,decode_agg;double kernel_ms=0,prefill_kernel_ms=0,decode_kernel_ms=0;
        std::ofstream f(csv);f<<"phase,layer,kind,tensor,type,cols,rows,batch,calls,ms\n";
        auto record=[&](const char*phase,const std::vector<memvanta::KernelProfileEntry>&entries,Agg&phase_agg,double&phase_ms){
            for(const auto&e:entries){const auto&t=m.gguf().tensor(e.tensor);f<<phase<<','<<e.layer<<','<<memvanta::kernel_profile_kind_name(e.kind)<<','<<e.tensor<<','<<memvanta::ggml_type_name(t.type)<<','<<t.ne(0)<<','<<t.ne(1)<<','<<e.batch<<','<<e.calls<<','<<std::fixed<<std::setprecision(6)<<e.ms<<'\n';phase_agg[{e.layer,e.kind}]+=e.ms;agg[{e.layer,e.kind}]+=e.ms;phase_ms+=e.ms;kernel_ms+=e.ms;}
        };
        record("prefill",prefill_entries,prefill_agg,prefill_kernel_ms);record("decode",decode_entries,decode_agg,decode_kernel_ms);
        const double prefill_ms=std::chrono::duration<double,std::milli>(t1-t0).count();const double decode_total_ms=std::chrono::duration<double,std::milli>(t2-t1).count();const double model_total_ms=std::chrono::duration<double,std::milli>(t2-t0).count();
        auto categories=[](const Agg&a){std::array<double,4> z{};for(const auto&[k,v]:a){switch(k.second){case memvanta::KernelProfileKind::QProj:case memvanta::KernelProfileKind::KProj:case memvanta::KernelProfileKind::VProj:z[0]+=v;break;case memvanta::KernelProfileKind::OProj:z[1]+=v;break;case memvanta::KernelProfileKind::FfnGate:case memvanta::KernelProfileKind::FfnUp:case memvanta::KernelProfileKind::FfnDown:z[2]+=v;break;case memvanta::KernelProfileKind::Output:z[3]+=v;break;default:break;}}return z;};
        const auto all=categories(agg),pre=categories(prefill_agg),dec=categories(decode_agg);const double qkv=all[0],attn_proj=all[1],ffn=all[2],output=all[3];
        const double residual=std::max(0.0,model_total_ms-kernel_ms-sampling_ms);
        std::cout<<"# MemVanta trained-model layer profile\n";
        std::cout<<"prefill_ms="<<prefill_ms<<" decode_total_ms="<<decode_total_ms<<" sampling_ms="<<sampling_ms<<"\n";
        if(c0&&c1>=c0&&c2>=c1){
            const double prefill_cpt=ids.empty()?0.0:double(c1-c0)/double(ids.size());
            const double decode_cpt=gen_n?double(c2-c1)/double(gen_n):0.0;
            std::cout<<"prefill_cycles_per_token="<<std::fixed<<std::setprecision(2)<<prefill_cpt<<" decode_cycles_per_token="<<decode_cpt<<"\n";
        }
        std::cout<<"qkv_projection_ms="<<qkv<<" attention_output_projection_ms="<<attn_proj<<" ffn_gemm_ms="<<ffn<<" output_head_ms="<<output<<"\n";
        std::cout<<"prefill_qkv_projection_ms="<<pre[0]<<" prefill_attention_output_projection_ms="<<pre[1]<<" prefill_ffn_gemm_ms="<<pre[2]<<" prefill_output_head_ms="<<pre[3]<<" prefill_kernel_ms="<<prefill_kernel_ms<<"\n";
        std::cout<<"decode_qkv_projection_ms="<<dec[0]<<" decode_attention_output_projection_ms="<<dec[1]<<" decode_ffn_gemm_ms="<<dec[2]<<" decode_output_head_ms="<<dec[3]<<" decode_kernel_ms="<<decode_kernel_ms<<"\n";
        std::cout<<"non_gemm_core_ms="<<residual<<" (attention softmax/KV + RMSNorm + RoPE + elementwise/residual)\n";
        std::cout<<"layer,qkv_ms,o_proj_ms,ffn_ms,total_projection_ms,pct_kernel_ms,projection_ms_per_token\n";
        const double measured_tokens=static_cast<double>(ids.size()+gen_n);
        for(std::size_t li=0;li<m.config().n_layer;++li){double lq=0,lo=0,lf=0;for(auto&[k,v]:agg)if(k.first==static_cast<int>(li)){switch(k.second){case memvanta::KernelProfileKind::QProj:case memvanta::KernelProfileKind::KProj:case memvanta::KernelProfileKind::VProj:lq+=v;break;case memvanta::KernelProfileKind::OProj:lo+=v;break;case memvanta::KernelProfileKind::FfnGate:case memvanta::KernelProfileKind::FfnUp:case memvanta::KernelProfileKind::FfnDown:lf+=v;break;default:break;}}const double total=lq+lo+lf;const double pct=kernel_ms>0?100.0*total/kernel_ms:0.0;const double per_token=measured_tokens>0?total/measured_tokens:0.0;std::cout<<li<<','<<lq<<','<<lo<<','<<lf<<','<<total<<','<<pct<<','<<per_token<<'\n';}
        std::cout<<"profile_csv="<<csv<<"\n";return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}
}