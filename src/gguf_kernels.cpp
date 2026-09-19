#include "memvanta/gguf_kernels.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(MEMVANTA_USE_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace memvanta {
namespace {
#pragma pack(push,1)
struct GgufBlockQ4_0 { std::uint16_t d; std::uint8_t qs[16]; };
struct GgufBlockQ8_0 { std::uint16_t d; std::int8_t qs[32]; };
// GGML Q6_K: 256 weights per super-block. Layout matches upstream ggml:
// 128 low-nibble bytes, 64 high-2-bit bytes, 16 signed sub-block scales,
// then one fp16 super-block scale.
struct GgufBlockQ6_K { std::uint8_t ql[128]; std::uint8_t qh[64]; std::int8_t scales[16]; std::uint16_t d; };
#pragma pack(pop)
static_assert(sizeof(GgufBlockQ4_0)==18);
static_assert(sizeof(GgufBlockQ8_0)==34);
static_assert(sizeof(GgufBlockQ6_K)==210);

using Clock = std::chrono::steady_clock;
std::atomic<bool> g_profile_enabled{false};
std::mutex g_profile_mu;
std::vector<KernelProfileEntry> g_profile_entries;

KernelProfileKind classify_tensor(const std::string& name) {
    if (name.find(".attn_q.") != std::string::npos) return KernelProfileKind::QProj;
    if (name.find(".attn_k.") != std::string::npos) return KernelProfileKind::KProj;
    if (name.find(".attn_v.") != std::string::npos) return KernelProfileKind::VProj;
    if (name.find(".attn_output.") != std::string::npos) return KernelProfileKind::OProj;
    if (name.find(".ffn_gate.") != std::string::npos) return KernelProfileKind::FfnGate;
    if (name.find(".ffn_up.") != std::string::npos) return KernelProfileKind::FfnUp;
    if (name.find(".ffn_down.") != std::string::npos) return KernelProfileKind::FfnDown;
    if (name == "output.weight") return KernelProfileKind::Output;
    return KernelProfileKind::Other;
}

bool is_ffn_tensor(const std::string& name) {
    return name.find(".ffn_gate.") != std::string::npos ||
           name.find(".ffn_up.") != std::string::npos ||
           name.find(".ffn_down.") != std::string::npos;
}

int tensor_layer(const std::string& name) {
    if (name.rfind("blk.",0) != 0) return -1;
    auto p = name.find('.',4);
    if (p == std::string::npos) return -1;
    try { return std::stoi(name.substr(4,p-4)); } catch (...) { return -1; }
}

void profile_add(const GgufTensor& t, std::size_t batch, double ms) {
    if (!g_profile_enabled.load(std::memory_order_relaxed)) return;
    KernelProfileEntry e{tensor_layer(t.name),classify_tensor(t.name),t.name,1,batch,ms};
    std::lock_guard<std::mutex> lock(g_profile_mu);
    for (auto& x : g_profile_entries) {
        if (x.layer == e.layer && x.kind == e.kind && x.tensor == e.tensor && x.batch == e.batch) {
            x.calls += 1; x.ms += e.ms; return;
        }
    }
    g_profile_entries.push_back(std::move(e));
}

template<class Fn>
void parallel_rows(std::size_t rows, unsigned threads, WorkerPool* pool, Fn fn) {
    threads=std::max(1u,threads);threads=std::min<unsigned>(threads,static_cast<unsigned>(std::max<std::size_t>(rows,1)));
    if(pool && pool->size()==threads){pool->parallel_for(rows,fn);return;}
    if(threads==1){fn(0,rows);return;}
#if defined(MEMVANTA_USE_OPENMP)
    const std::size_t step=(rows+threads-1)/threads;
    #pragma omp parallel num_threads(threads)
    {
        const unsigned tid=static_cast<unsigned>(omp_get_thread_num());
        const std::size_t a=tid*step,b=std::min(rows,a+step);if(a<b)fn(a,b);
    }
#else
    std::vector<std::thread> tmp;const std::size_t step=(rows+threads-1)/threads;
    for(unsigned tid=0;tid<threads;++tid){auto a=tid*step,b=std::min(rows,a+step);if(a<b)tmp.emplace_back(fn,a,b);}for(auto&th:tmp)th.join();
#endif
}

inline float dot_q4_0_fp32(const GgufBlockQ4_0* blocks,const float*x,std::size_t n){
    const std::size_t nb=n/32;float sum=0.0f;
    for(std::size_t b=0;b<nb;++b){const float d=fp16_to_fp32(blocks[b].d);
#if defined(__AVX2__)
        const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[b].qs));
        const __m128i mask=_mm_set1_epi8(0x0f),bias=_mm_set1_epi8(8);
        const __m128i qlo=_mm_sub_epi8(_mm_and_si128(packed,mask),bias);
        const __m128i qhi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed,4),mask),bias);
        __m256 acc=_mm256_setzero_ps();
        for(int k=0;k<16;k+=8){auto qi=_mm256_cvtepi8_epi32(_mm_srli_si128(qlo,k));acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(qi),_mm256_loadu_ps(x+b*32+k),acc);}
        for(int k=0;k<16;k+=8){auto qi=_mm256_cvtepi8_epi32(_mm_srli_si128(qhi,k));acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(qi),_mm256_loadu_ps(x+b*32+16+k),acc);}
        alignas(32) float tmp[8];_mm256_store_ps(tmp,acc);float s=0;for(float v:tmp)s+=v;sum+=d*s;
#else
        float s=0;for(std::size_t i=0;i<16;++i){s+=(int(blocks[b].qs[i]&15)-8)*x[b*32+i];s+=(int(blocks[b].qs[i]>>4)-8)*x[b*32+16+i];}sum+=d*s;
#endif
    }return sum;
}

inline float dot_q8_0_fp32(const GgufBlockQ8_0* blocks,const float*x,std::size_t n){
    const std::size_t nb=n/32;float sum=0;
    for(std::size_t b=0;b<nb;++b){float d=fp16_to_fp32(blocks[b].d);
#if defined(__AVX2__)
        __m256 acc=_mm256_setzero_ps();for(int k=0;k<32;k+=8){auto q8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(blocks[b].qs+k));auto qi=_mm256_cvtepi8_epi32(q8);acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(qi),_mm256_loadu_ps(x+b*32+k),acc);}alignas(32)float tmp[8];_mm256_store_ps(tmp,acc);float s=0;for(float v:tmp)s+=v;sum+=d*s;
#else
        float s=0;for(std::size_t k=0;k<32;++k)s+=blocks[b].qs[k]*x[b*32+k];sum+=d*s;
#endif
    }return sum;
}

inline int q6_k_value(const GgufBlockQ6_K& b,std::size_t i){
    // This scalar mapping is the direct equivalent of gguf-py's Q6_K
    // dequantization reshape/bit-unpack sequence.
    const std::size_t row32=i/32;
    const std::size_t j=i%32;
    const std::size_t group=row32/4;
    const std::size_t local=row32%4;
    const std::size_t ql_index=group*64+(local%2)*32+j;
    const unsigned ql_shift=local<2?0u:4u;
    const std::size_t qh_index=group*32+j;
    const unsigned qh_shift=static_cast<unsigned>(local*2);
    const int lo=(b.ql[ql_index]>>ql_shift)&0x0f;
    const int hi=(b.qh[qh_index]>>qh_shift)&0x03;
    return (lo|(hi<<4))-32;
}

inline float dot_q6_k_fp32(const GgufBlockQ6_K* blocks,const float*x,std::size_t n){
    if(n%256)throw std::runtime_error("Q6_K row requires 256-element alignment");
    const std::size_t nb=n/256;double sum=0.0;
    for(std::size_t bi=0;bi<nb;++bi){
        const auto& b=blocks[bi];const float d=fp16_to_fp32(b.d);const float* xv=x+bi*256;
        for(std::size_t i=0;i<256;++i){const float w=d*float(b.scales[i/16])*float(q6_k_value(b,i));sum+=double(w)*xv[i];}
    }
    return static_cast<float>(sum);
}

inline int dot_i8_16(const std::int8_t* a,const std::int8_t* b){
#if defined(__AVX2__)
    __m128i aa=_mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    __m128i bb=_mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    __m256i a16=_mm256_cvtepi8_epi16(aa),b16=_mm256_cvtepi8_epi16(bb);
    __m256i p32=_mm256_madd_epi16(a16,b16);
    __m128i lo=_mm256_castsi256_si128(p32),hi=_mm256_extracti128_si256(p32,1);__m128i v=_mm_add_epi32(lo,hi);v=_mm_hadd_epi32(v,v);v=_mm_hadd_epi32(v,v);return _mm_cvtsi128_si32(v);
#else
    int s=0;for(int i=0;i<16;++i)s+=int(a[i])*int(b[i]);return s;
#endif
}

struct Q8Vector {std::vector<std::int8_t> q;std::vector<float> d;};
Q8Vector quantize_q8(const float* x,std::size_t cols){
    if(cols%32)throw std::runtime_error("Q8 activation quantization requires cols multiple of 32");
    Q8Vector a;a.q.resize(cols);a.d.resize(cols/32);
    for(std::size_t bi=0;bi<cols/32;++bi){const float*z=x+bi*32;float mx=0.0f;
#if defined(__AVX2__)
        __m256 m=_mm256_setzero_ps();const __m256 sign=_mm256_set1_ps(-0.0f);for(int k=0;k<32;k+=8){auto v=_mm256_andnot_ps(sign,_mm256_loadu_ps(z+k));m=_mm256_max_ps(m,v);}alignas(32)float mm[8];_mm256_store_ps(mm,m);for(float v:mm)mx=std::max(mx,v);
#else
        for(int k=0;k<32;++k)mx=std::max(mx,std::abs(z[k]));
#endif
        float d=mx>0?mx/127.0f:1.0f,inv=1.0f/d;a.d[bi]=d;for(int k=0;k<32;++k)a.q[bi*32+k]=static_cast<std::int8_t>(std::clamp(std::lround(z[k]*inv),-127l,127l));
    }return a;
}

struct Q8ActBatch {std::size_t batch{},cols{},blocks{};std::vector<std::int8_t>q;std::vector<float>d;};
Q8ActBatch quantize_activations_q8(const float*x,std::size_t batch,std::size_t cols){
    Q8ActBatch a;a.batch=batch;a.cols=cols;a.blocks=cols/32;a.q.resize(batch*cols);a.d.resize(batch*a.blocks);
    for(std::size_t b=0;b<batch;++b){auto v=quantize_q8(x+b*cols,cols);std::memcpy(a.q.data()+b*cols,v.q.data(),cols);std::memcpy(a.d.data()+b*a.blocks,v.d.data(),a.blocks*sizeof(float));}return a;
}

inline int dot_q4_q8_block(const GgufBlockQ4_0&w,const std::int8_t*aq){
    alignas(16)std::int8_t lo[16],hi[16];
#if defined(__AVX2__)
    const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));const __m128i mask=_mm_set1_epi8(0x0f),bias=_mm_set1_epi8(8);_mm_store_si128(reinterpret_cast<__m128i*>(lo),_mm_sub_epi8(_mm_and_si128(packed,mask),bias));_mm_store_si128(reinterpret_cast<__m128i*>(hi),_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed,4),mask),bias));
#else
    for(int i=0;i<16;++i){lo[i]=std::int8_t((w.qs[i]&15)-8);hi[i]=std::int8_t((w.qs[i]>>4)-8);}
#endif
    return dot_i8_16(lo,aq)+dot_i8_16(hi,aq+16);
}
inline float dot_q4_q8(const GgufBlockQ4_0*w,const std::int8_t*aq,const float*ad,std::size_t nb){float sum=0;for(std::size_t bi=0;bi<nb;++bi)sum+=fp16_to_fp32(w[bi].d)*ad[bi]*float(dot_q4_q8_block(w[bi],aq+bi*32));return sum;}

inline void q4_row_batch4_fp32(const GgufBlockQ4_0*w,const float*x,std::size_t stride,std::size_t nb,float out[4]){
#if defined(__AVX2__)
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();const __m128i mask=_mm_set1_epi8(0x0f),bias=_mm_set1_epi8(8);
    for(std::size_t bi=0;bi<nb;++bi){const float d=fp16_to_fp32(w[bi].d);const __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w[bi].qs));const __m128i lo=_mm_sub_epi8(_mm_and_si128(packed,mask),bias),hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed,4),mask),bias);for(int h=0;h<2;++h){const __m128i src=h?hi:lo;for(int k=0;k<16;k+=8){__m256 q=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(src,k))),_mm256_set1_ps(d));std::size_t off=bi*32+h*16+k;a0=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off),a0);a1=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+stride+off),a1);a2=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+2*stride+off),a2);a3=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+3*stride+off),a3);}}}
    alignas(32)float t[8];auto hs=[&](const __m256&v){_mm256_store_ps(t,v);return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];};out[0]=hs(a0);out[1]=hs(a1);out[2]=hs(a2);out[3]=hs(a3);
#else
    for(int j=0;j<4;++j)out[j]=dot_q4_0_fp32(w,x+j*stride,nb*32);
#endif
}

inline void q8_row_batch4_fp32(const GgufBlockQ8_0*w,const float*x,std::size_t stride,std::size_t nb,float out[4]){
#if defined(__AVX2__)
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
    for(std::size_t bi=0;bi<nb;++bi){const __m256 ds=_mm256_set1_ps(fp16_to_fp32(w[bi].d));for(int k=0;k<32;k+=8){__m128i q8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w[bi].qs+k));__m256 q=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)),ds);std::size_t off=bi*32+k;a0=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+off),a0);a1=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+stride+off),a1);a2=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+2*stride+off),a2);a3=_mm256_fmadd_ps(q,_mm256_loadu_ps(x+3*stride+off),a3);}}
    alignas(32)float t[8];auto hs=[&](const __m256&v){_mm256_store_ps(t,v);return t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];};out[0]=hs(a0);out[1]=hs(a1);out[2]=hs(a2);out[3]=hs(a3);
#else
    for(int j=0;j<4;++j)out[j]=dot_q8_0_fp32(w,x+j*stride,nb*32);
#endif
}

inline void q4_row_batch8_fp32(const GgufBlockQ4_0*w,const float*x,std::size_t stride,std::size_t nb,float out[8]){
    float a[4],b[4];q4_row_batch4_fp32(w,x,stride,nb,a);q4_row_batch4_fp32(w,x+4*stride,stride,nb,b);for(int i=0;i<4;++i){out[i]=a[i];out[i+4]=b[i];}
}
inline void q8_row_batch8_fp32(const GgufBlockQ8_0*w,const float*x,std::size_t stride,std::size_t nb,float out[8]){
    float a[4],b[4];q8_row_batch4_fp32(w,x,stride,nb,a);q8_row_batch4_fp32(w,x+4*stride,stride,nb,b);for(int i=0;i<4;++i){out[i]=a[i];out[i+4]=b[i];}
}
} // namespace

void enable_kernel_profiling(bool enabled){g_profile_enabled.store(enabled,std::memory_order_relaxed);}
void reset_kernel_profile(){std::lock_guard<std::mutex>lock(g_profile_mu);g_profile_entries.clear();}
std::vector<KernelProfileEntry> kernel_profile_snapshot(){std::lock_guard<std::mutex>lock(g_profile_mu);return g_profile_entries;}
const char* kernel_profile_kind_name(KernelProfileKind k){switch(k){case KernelProfileKind::QProj:return "q_proj";case KernelProfileKind::KProj:return "k_proj";case KernelProfileKind::VProj:return "v_proj";case KernelProfileKind::OProj:return "o_proj";case KernelProfileKind::FfnGate:return "ffn_gate";case KernelProfileKind::FfnUp:return "ffn_up";case KernelProfileKind::FfnDown:return "ffn_down";case KernelProfileKind::Output:return "output";case KernelProfileKind::Other:return "other";}return "other";}

void tensor_vector_to_f32(const GgufFile&file,const GgufTensor&t,float*out,std::size_t n){if(t.elements()!=n)throw std::runtime_error("tensor vector size mismatch: "+t.name);const std::byte*p=file.tensor_data(t);if(t.type==GgmlType::F32){std::memcpy(out,p,n*sizeof(float));return;}if(t.type==GgmlType::F16){auto*q=reinterpret_cast<const std::uint16_t*>(p);for(std::size_t i=0;i<n;++i)out[i]=fp16_to_fp32(q[i]);return;}throw std::runtime_error("unsupported vector tensor type "+ggml_type_name(t.type)+": "+t.name);}

void tensor_read_row_f32(const GgufFile&file,const GgufTensor&t,std::size_t row,float*out,std::size_t cols){if(t.dims.size()<2||t.ne(0)!=cols||row>=t.ne(1))throw std::runtime_error("tensor row shape mismatch: "+t.name);const std::byte*p=file.tensor_data(t);if(t.type==GgmlType::F32){std::memcpy(out,reinterpret_cast<const float*>(p)+row*cols,cols*sizeof(float));return;}if(t.type==GgmlType::F16){auto*q=reinterpret_cast<const std::uint16_t*>(p)+row*cols;for(std::size_t i=0;i<cols;++i)out[i]=fp16_to_fp32(q[i]);return;}if(t.type==GgmlType::Q4_0){auto*b=reinterpret_cast<const GgufBlockQ4_0*>(p)+row*(cols/32);for(std::size_t bi=0;bi<cols/32;++bi){float d=fp16_to_fp32(b[bi].d);for(std::size_t i=0;i<16;++i){out[bi*32+i]=d*(int(b[bi].qs[i]&15)-8);out[bi*32+16+i]=d*(int(b[bi].qs[i]>>4)-8);}}return;}if(t.type==GgmlType::Q8_0){auto*b=reinterpret_cast<const GgufBlockQ8_0*>(p)+row*(cols/32);for(std::size_t bi=0;bi<cols/32;++bi){float d=fp16_to_fp32(b[bi].d);for(std::size_t i=0;i<32;++i)out[bi*32+i]=d*b[bi].qs[i];}return;}if(t.type==GgmlType::Q6_K){if(cols%256)throw std::runtime_error("Q6_K embedding row not block aligned: "+t.name);auto*b=reinterpret_cast<const GgufBlockQ6_K*>(p)+row*(cols/256);for(std::size_t bi=0;bi<cols/256;++bi){float d=fp16_to_fp32(b[bi].d);for(std::size_t i=0;i<256;++i)out[bi*256+i]=d*float(b[bi].scales[i/16])*float(q6_k_value(b[bi],i));}return;}throw std::runtime_error("unsupported embedding row type "+ggml_type_name(t.type)+": "+t.name);}

float dot_f32_simd(const float*a,const float*b,std::size_t n){
#if defined(__AVX2__)
    __m256 acc=_mm256_setzero_ps();std::size_t i=0;for(;i+8<=n;i+=8)acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);alignas(32)float t[8];_mm256_store_ps(t,acc);float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];for(;i<n;++i)s+=a[i]*b[i];return s;
#else
    float s=0;for(std::size_t i=0;i<n;++i)s+=a[i]*b[i];return s;
#endif
}
void axpy_f32_simd(float alpha,const float*x,float*y,std::size_t n){
#if defined(__AVX2__)
    __m256 av=_mm256_set1_ps(alpha);std::size_t i=0;for(;i+8<=n;i+=8){auto xv=_mm256_loadu_ps(x+i),yv=_mm256_loadu_ps(y+i);_mm256_storeu_ps(y+i,_mm256_fmadd_ps(av,xv,yv));}for(;i<n;++i)y[i]+=alpha*x[i];
#else
    for(std::size_t i=0;i<n;++i)y[i]+=alpha*x[i];
#endif
}
std::uint16_t fp32_to_fp16(float f){std::uint32_t x;std::memcpy(&x,&f,4);std::uint32_t sign=(x>>16)&0x8000u,mant=x&0x7fffffu;int exp=int((x>>23)&0xff)-127+15;if(exp<=0){if(exp<-10)return static_cast<std::uint16_t>(sign);mant=(mant|0x800000u)>>(1-exp);return static_cast<std::uint16_t>(sign+((mant+0x1000u)>>13));}if(exp>=31)return static_cast<std::uint16_t>(sign|0x7c00u);return static_cast<std::uint16_t>(sign|(std::uint32_t(exp)<<10)|((mant+0x1000u)>>13));}

void tensor_matvec(const GgufFile&file,const GgufTensor&t,const float*x,float*y,unsigned threads,WorkerPool*pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("matvec requires rank-2 tensor: "+t.name);const std::size_t cols=t.ne(0),rows=t.ne(1);const std::byte*p=file.tensor_data(t);
    if(t.type==GgmlType::Q4_0){if(cols%32)throw std::runtime_error("Q4_0 matrix row not block aligned: "+t.name);auto a=quantize_q8(x,cols);auto*A=reinterpret_cast<const GgufBlockQ4_0*>(p);auto nb=cols/32;parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r)y[r]=dot_q4_q8(A+r*nb,a.q.data(),a.d.data(),nb);});}
    else if(t.type==GgmlType::Q8_0){auto*A=reinterpret_cast<const GgufBlockQ8_0*>(p);auto nb=cols/32;parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r)y[r]=dot_q8_0_fp32(A+r*nb,x,cols);});}
    else if(t.type==GgmlType::Q6_K){if(cols%256)throw std::runtime_error("Q6_K matrix row not block aligned: "+t.name);auto*A=reinterpret_cast<const GgufBlockQ6_K*>(p);auto nb=cols/256;parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r)y[r]=dot_q6_k_fp32(A+r*nb,x,cols);});}
    else if(t.type==GgmlType::F32){auto*A=reinterpret_cast<const float*>(p);parallel_rows(rows,threads,pool,[&](std::size_t a,std::size_t b){for(std::size_t r=a;r<b;++r)y[r]=dot_f32_simd(A+r*cols,x,cols);});}
    else if(t.type==GgmlType::F16){auto*A=reinterpret_cast<const std::uint16_t*>(p);parallel_rows(rows,threads,pool,[&](std::size_t a,std::size_t b){for(std::size_t r=a;r<b;++r){double s=0;for(std::size_t c=0;c<cols;++c)s+=double(fp16_to_fp32(A[r*cols+c]))*x[c];y[r]=float(s);}});}
    else throw std::runtime_error("unsupported matvec tensor type "+ggml_type_name(t.type)+": "+t.name);
    profile_add(t,1,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

void tensor_matmul_batch(const GgufFile&file,const GgufTensor&t,const float*x,float*y,std::size_t batch,unsigned threads,WorkerPool*pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("matmul batch requires rank-2 tensor: "+t.name);const std::size_t cols=t.ne(0),rows=t.ne(1);const std::byte*p=file.tensor_data(t);const std::size_t jobs=batch*rows;
    parallel_rows(jobs,threads,pool,[&](std::size_t a,std::size_t b){for(std::size_t j=a;j<b;++j){std::size_t bi=j/rows,r=j%rows;const float*xv=x+bi*cols;float v=0;if(t.type==GgmlType::Q4_0)v=dot_q4_0_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*(cols/32),xv,cols);else if(t.type==GgmlType::Q8_0)v=dot_q8_0_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*(cols/32),xv,cols);else if(t.type==GgmlType::Q6_K){if(cols%256)throw std::runtime_error("Q6_K batch matrix row not block aligned: "+t.name);v=dot_q6_k_fp32(reinterpret_cast<const GgufBlockQ6_K*>(p)+r*(cols/256),xv,cols);}else if(t.type==GgmlType::F32)v=dot_f32_simd(reinterpret_cast<const float*>(p)+r*cols,xv,cols);else if(t.type==GgmlType::F16){auto*A=reinterpret_cast<const std::uint16_t*>(p)+r*cols;double s=0;for(std::size_t c=0;c<cols;++c)s+=double(fp16_to_fp32(A[c]))*xv[c];v=float(s);}else throw std::runtime_error("unsupported batch matmul tensor type "+ggml_type_name(t.type)+": "+t.name);y[bi*rows+r]=v;}});
    profile_add(t,batch,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

void tensor_matmul_batch_v06(const GgufFile&file,const GgufTensor&t,const float*x,float*y,std::size_t batch,unsigned threads,WorkerPool*pool){
    const auto t0=Clock::now();if(t.dims.size()<2)throw std::runtime_error("hybrid GEMM requires rank-2 tensor: "+t.name);const std::size_t cols=t.ne(0),rows=t.ne(1);
    if(batch<2||(t.type!=GgmlType::Q4_0&&t.type!=GgmlType::Q8_0)||cols%32){tensor_matmul_batch(file,t,x,y,batch,threads,pool);return;}
    const std::byte*p=file.tensor_data(t);const std::size_t nb=cols/32;
    if(const char*e=std::getenv("MEMVANTA_FORCE_Q8_ACT");e&&*e=='1'){
        auto a=quantize_activations_q8(x,batch,cols);parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r){for(std::size_t b=0;b<batch;++b){if(t.type==GgmlType::Q4_0)y[b*rows+r]=dot_q4_q8(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,a.q.data()+b*cols,a.d.data()+b*nb,nb);else y[b*rows+r]=dot_q8_0_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols);}}});
    } else {
        const bool ffn=is_ffn_tensor(t.name);parallel_rows(rows,threads,pool,[&](std::size_t r0,std::size_t r1){for(std::size_t r=r0;r<r1;++r){std::size_t b=0;if(ffn){for(;b+8<=batch;b+=8){float o[8];if(t.type==GgmlType::Q4_0)q4_row_batch8_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch8_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<8;++j)y[(b+j)*rows+r]=o[j];}}else{for(;b+4<=batch;b+=4){float o[4];if(t.type==GgmlType::Q4_0)q4_row_batch4_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols,nb,o);else q8_row_batch4_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols,nb,o);for(int j=0;j<4;++j)y[(b+j)*rows+r]=o[j];}}for(;b<batch;++b){if(t.type==GgmlType::Q4_0)y[b*rows+r]=dot_q4_0_fp32(reinterpret_cast<const GgufBlockQ4_0*>(p)+r*nb,x+b*cols,cols);else y[b*rows+r]=dot_q8_0_fp32(reinterpret_cast<const GgufBlockQ8_0*>(p)+r*nb,x+b*cols,cols);}}});
    }
    profile_add(t,batch,std::chrono::duration<double,std::milli>(Clock::now()-t0).count());
}

} // namespace memvanta
