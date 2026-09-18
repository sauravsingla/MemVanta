#include "memvanta/llama_model.hpp"
#include "memvanta/gguf_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace memvanta {

KVCacheType parse_kv_cache_type(const std::string&s){if(s=="f32")return KVCacheType::F32;if(s=="f16")return KVCacheType::F16;if(s=="q8"||s=="q8_0")return KVCacheType::Q8;throw std::runtime_error("unknown KV cache type: "+s);}
const char* kv_cache_type_name(KVCacheType t){switch(t){case KVCacheType::F32:return "f32";case KVCacheType::F16:return "f16";case KVCacheType::Q8:return "q8";}return "?";}
PagedKVCache::Page::Page(std::size_t elems,std::size_t tokens,KVCacheType type){if(type==KVCacheType::F32){k32.resize(elems);v32.resize(elems);}else if(type==KVCacheType::F16){k16.resize(elems);v16.resize(elems);}else{k8.resize(elems);v8.resize(elems);ks.resize(tokens);vs.resize(tokens);}}
PagedKVCache::PagedKVCache(std::size_t d,std::size_t p,KVCacheType type):kv_dim_(d),page_tokens_(p?p:128),type_(type){}
void PagedKVCache::reset(){pages_.clear();}
PagedKVCache::Page& PagedKVCache::ensure(std::size_t p){if(p>=pages_.size())pages_.resize(p+1);if(!pages_[p])pages_[p]=std::make_unique<Page>(page_tokens_*kv_dim_,page_tokens_,type_);return *pages_[p];}
void PagedKVCache::write(std::size_t pos,const float*k,const float*v){auto p=pos/page_tokens_,i=pos%page_tokens_;auto&pg=ensure(p);auto off=i*kv_dim_;if(type_==KVCacheType::F32){std::memcpy(pg.k32.data()+off,k,kv_dim_*sizeof(float));std::memcpy(pg.v32.data()+off,v,kv_dim_*sizeof(float));}else if(type_==KVCacheType::F16){for(std::size_t d=0;d<kv_dim_;++d){pg.k16[off+d]=fp32_to_fp16(k[d]);pg.v16[off+d]=fp32_to_fp16(v[d]);}}else{float km=0,vm=0;for(std::size_t d=0;d<kv_dim_;++d){km=std::max(km,std::abs(k[d]));vm=std::max(vm,std::abs(v[d]));}float ks=km>0?km/127.0f:1.0f,vs=vm>0?vm/127.0f:1.0f;pg.ks[i]=ks;pg.vs[i]=vs;for(std::size_t d=0;d<kv_dim_;++d){pg.k8[off+d]=static_cast<std::int8_t>(std::clamp(std::lround(k[d]/ks),-127l,127l));pg.v8[off+d]=static_cast<std::int8_t>(std::clamp(std::lround(v[d]/vs),-127l,127l));}}}
float PagedKVCache::dot_key(std::size_t pos,std::size_t offset,const float*q,std::size_t n)const{
    auto p=pos/page_tokens_,i=pos%page_tokens_;if(p>=pages_.size()||!pages_[p])throw std::runtime_error("KV key page missing");auto&pg=*pages_[p];auto off=i*kv_dim_+offset;
    if(type_==KVCacheType::F32)return dot_f32_simd(q,pg.k32.data()+off,n);
    if(type_==KVCacheType::F16){
#if defined(__AVX2__) && defined(__F16C__)
        __m256 acc=_mm256_setzero_ps();std::size_t d=0;for(;d+8<=n;d+=8){__m128i h=_mm_loadu_si128(reinterpret_cast<const __m128i*>(pg.k16.data()+off+d));__m256 v=_mm256_cvtph_ps(h);acc=_mm256_fmadd_ps(_mm256_loadu_ps(q+d),v,acc);}alignas(32)float tmp[8];_mm256_store_ps(tmp,acc);float sum=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];for(;d<n;++d)sum+=q[d]*fp16_to_fp32(pg.k16[off+d]);return sum;
#else
        float sum=0;for(std::size_t d=0;d<n;++d)sum+=q[d]*fp16_to_fp32(pg.k16[off+d]);return sum;
#endif
    }
    float sc=pg.ks[i];
#if defined(__AVX2__)
    __m256 acc=_mm256_setzero_ps();std::size_t d=0;for(;d+8<=n;d+=8){__m128i q8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pg.k8.data()+off+d));__m256i qi=_mm256_cvtepi8_epi32(q8);acc=_mm256_fmadd_ps(_mm256_loadu_ps(q+d),_mm256_cvtepi32_ps(qi),acc);}alignas(32)float tmp[8];_mm256_store_ps(tmp,acc);float sum=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];for(;d<n;++d)sum+=q[d]*float(pg.k8[off+d]);return sum*sc;
#else
    float sum=0;for(std::size_t d=0;d<n;++d)sum+=q[d]*float(pg.k8[off+d]);return sum*sc;
#endif
}
void PagedKVCache::add_value(std::size_t pos,std::size_t offset,float alpha,float*out,std::size_t n)const{
    auto p=pos/page_tokens_,i=pos%page_tokens_;if(p>=pages_.size()||!pages_[p])throw std::runtime_error("KV value page missing");auto&pg=*pages_[p];auto off=i*kv_dim_+offset;if(type_==KVCacheType::F32){axpy_f32_simd(alpha,pg.v32.data()+off,out,n);return;}
    if(type_==KVCacheType::F16){
#if defined(__AVX2__) && defined(__F16C__)
        __m256 av=_mm256_set1_ps(alpha);std::size_t d=0;for(;d+8<=n;d+=8){__m128i h=_mm_loadu_si128(reinterpret_cast<const __m128i*>(pg.v16.data()+off+d));__m256 v=_mm256_cvtph_ps(h);__m256 y=_mm256_loadu_ps(out+d);_mm256_storeu_ps(out+d,_mm256_fmadd_ps(av,v,y));}for(;d<n;++d)out[d]+=alpha*fp16_to_fp32(pg.v16[off+d]);return;
#else
        for(std::size_t d=0;d<n;++d)out[d]+=alpha*fp16_to_fp32(pg.v16[off+d]);return;
#endif
    }
    float sc=alpha*pg.vs[i];
#if defined(__AVX2__)
    __m256 sv=_mm256_set1_ps(sc);std::size_t d=0;for(;d+8<=n;d+=8){__m128i q8=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pg.v8.data()+off+d));__m256i qi=_mm256_cvtepi8_epi32(q8);__m256 y=_mm256_loadu_ps(out+d);_mm256_storeu_ps(out+d,_mm256_fmadd_ps(sv,_mm256_cvtepi32_ps(qi),y));}for(;d<n;++d)out[d]+=sc*float(pg.v8[off+d]);
#else
    for(std::size_t d=0;d<n;++d)out[d]+=sc*float(pg.v8[off+d]);
#endif
}
std::size_t PagedKVCache::bytes_allocated()const{std::size_t n=0;for(auto&p:pages_)if(p)n+=p->k32.size()*4+p->v32.size()*4+p->k16.size()*2+p->v16.size()*2+p->k8.size()+p->v8.size()+p->ks.size()*4+p->vs.size()*4;return n;}


Sampler::Sampler(SamplerConfig c):c_(c),rng_(c.seed){}
int Sampler::sample(const std::vector<float>& logits){if(logits.empty())throw std::runtime_error("empty logits");if(c_.temperature<=0)return static_cast<int>(std::max_element(logits.begin(),logits.end())-logits.begin());std::vector<int> idx(logits.size());std::iota(idx.begin(),idx.end(),0);std::size_t k=std::min(c_.top_k?c_.top_k:logits.size(),logits.size());std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){return logits[a]>logits[b];});float mx=logits[idx[0]]/c_.temperature;std::vector<double>w(k);double sum=0;for(std::size_t i=0;i<k;++i){w[i]=std::exp(logits[idx[i]]/c_.temperature-mx);sum+=w[i];}std::uniform_real_distribution<double>d(0,sum);double r=d(rng_);for(std::size_t i=0;i<k;++i){r-=w[i];if(r<=0)return idx[i];}return idx[k-1];}

namespace {
std::vector<std::string> make_byte_unicode(){
    std::vector<int> bs,cs;for(int i=33;i<=126;++i)bs.push_back(i);for(int i=161;i<=172;++i)bs.push_back(i);for(int i=174;i<=255;++i)bs.push_back(i);cs=bs;int n=0;for(int b=0;b<256;++b)if(std::find(bs.begin(),bs.end(),b)==bs.end()){bs.push_back(b);cs.push_back(256+n++);}std::vector<std::string> out(256);
    auto utf8=[](int cp){std::string s;if(cp<=0x7f)s.push_back(static_cast<char>(cp));else if(cp<=0x7ff){s.push_back(static_cast<char>(0xc0|(cp>>6)));s.push_back(static_cast<char>(0x80|(cp&63)));}else{s.push_back(static_cast<char>(0xe0|(cp>>12)));s.push_back(static_cast<char>(0x80|((cp>>6)&63)));s.push_back(static_cast<char>(0x80|(cp&63)));}return s;};
    for(std::size_t i=0;i<bs.size();++i)out[bs[i]]=utf8(cs[i]);return out;
}

struct Utf8Unit { std::size_t off{},len{}; std::uint32_t cp{}; };
Utf8Unit utf8_at(const std::string& s,std::size_t p){
    const auto c=static_cast<unsigned char>(s[p]);
    if(c<0x80)return{p,1,c};
    auto cont=[&](std::size_t i){return i<s.size() && (static_cast<unsigned char>(s[i])&0xc0)==0x80;};
    if((c&0xe0)==0xc0 && cont(p+1))return{p,2,static_cast<std::uint32_t>(((c&0x1f)<<6)|(static_cast<unsigned char>(s[p+1])&0x3f))};
    if((c&0xf0)==0xe0 && cont(p+1)&&cont(p+2))return{p,3,static_cast<std::uint32_t>(((c&0x0f)<<12)|((static_cast<unsigned char>(s[p+1])&0x3f)<<6)|(static_cast<unsigned char>(s[p+2])&0x3f))};
    if((c&0xf8)==0xf0 && cont(p+1)&&cont(p+2)&&cont(p+3))return{p,4,static_cast<std::uint32_t>(((c&0x07)<<18)|((static_cast<unsigned char>(s[p+1])&0x3f)<<12)|((static_cast<unsigned char>(s[p+2])&0x3f)<<6)|(static_cast<unsigned char>(s[p+3])&0x3f))};
    // Invalid UTF-8 is deliberately treated as one opaque byte. GPT-2 byte BPE and
    // SentencePiece byte fallback can still round-trip it exactly.
    return{p,1,c};
}
bool u_space(std::uint32_t cp){return cp==0x20||cp==0x09||cp==0x0a||cp==0x0d||cp==0x0b||cp==0x0c||cp==0x85||cp==0xa0||cp==0x1680||(cp>=0x2000&&cp<=0x200a)||cp==0x2028||cp==0x2029||cp==0x202f||cp==0x205f||cp==0x3000;}
bool u_digit(std::uint32_t cp){return cp>='0'&&cp<='9';}
bool u_letter(std::uint32_t cp){if(cp<128)return (cp>='A'&&cp<='Z')||(cp>='a'&&cp<='z');return !u_space(cp);}
std::string replace_all(std::string s,const std::string&a,const std::string&b){std::size_t p=0;while((p=s.find(a,p))!=std::string::npos){s.replace(p,a.size(),b);p+=b.size();}return s;}
}
std::string GgufTokenizer::gpt2_encode_bytes(const std::string&s){static const auto m=make_byte_unicode();std::string o;for(unsigned char c:s)o+=m[c];return o;}
std::string GgufTokenizer::gpt2_decode_bytes(const std::string&s){
    static const auto rev=[](){auto mm=make_byte_unicode();std::unordered_map<std::string,unsigned char>r;for(int i=0;i<256;++i)r[mm[i]]=static_cast<unsigned char>(i);return r;}();
    std::string o;for(std::size_t i=0;i<s.size();){auto u=utf8_at(s,i);std::string ch=s.substr(i,u.len);auto it=rev.find(ch);if(it!=rev.end())o.push_back(static_cast<char>(it->second));else o+=ch;i+=u.len;}return o;
}

GgufTokenizer::GgufTokenizer(const GgufFile&f){
    auto*a=f.get_string_array("tokenizer.ggml.tokens");
    if(!a)throw std::runtime_error("GGUF tokenizer tokens missing");
    tokens_=*a;
    model_=f.get_string("tokenizer.ggml.model").value_or("gpt2");
    pre_=f.get_string("tokenizer.ggml.pre").value_or("default");
    add_space_prefix_=f.get_bool("tokenizer.ggml.add_space_prefix").value_or(model_=="llama"||model_=="replit");
    bos_=static_cast<int>(f.get_u64("tokenizer.ggml.bos_token_id").value_or(std::numeric_limits<std::uint64_t>::max()));
    eos_=static_cast<int>(f.get_u64("tokenizer.ggml.eos_token_id").value_or(std::numeric_limits<std::uint64_t>::max()));
    unk_=static_cast<int>(f.get_u64("tokenizer.ggml.unknown_token_id").value_or(std::numeric_limits<std::uint64_t>::max()));
    if(auto*s=f.get_float_array("tokenizer.ggml.scores"))scores_=*s; else scores_.assign(tokens_.size(),0.0);
    if(auto*t=f.get_int_array("tokenizer.ggml.token_type"))token_types_=*t; else token_types_.assign(tokens_.size(),1);
    if(scores_.size()!=tokens_.size())scores_.resize(tokens_.size(),0.0);
    if(token_types_.size()!=tokens_.size())token_types_.resize(tokens_.size(),1);
    greedy_.reserve(tokens_.size());
    for(std::size_t i=0;i<tokens_.size();++i){
        token_to_id_[tokens_[i]]=static_cast<int>(i);
        greedy_.push_back({tokens_[i],static_cast<int>(i)});
        if(token_types_[i]==3||token_types_[i]==4)specials_.push_back({tokens_[i],static_cast<int>(i)});
        if(token_types_[i]==6 && tokens_[i].size()==6 && tokens_[i].rfind("<0x",0)==0 && tokens_[i].back()=='>'){
            try{int b=std::stoi(tokens_[i].substr(3,2),nullptr,16);if(b>=0&&b<=255)byte_token_id_[static_cast<unsigned char>(b)]=static_cast<int>(i);}catch(...){}
        }
    }
    std::sort(greedy_.begin(),greedy_.end(),[](auto&a,auto&b){return a.first.size()>b.first.size();});
    std::sort(specials_.begin(),specials_.end(),[](auto&a,auto&b){return a.first.size()>b.first.size();});
    {const auto bm=make_byte_unicode();byte_bpe_complete_=true;for(const auto&atom:bm)if(!token_to_id_.contains(atom)){byte_bpe_complete_=false;break;}}
    if(auto*m=f.get_string_array("tokenizer.ggml.merges")){
        for(std::size_t i=0;i<m->size();++i){const auto&z=(*m)[i];auto sp=z.find(' ');if(sp!=std::string::npos)merge_rank_[z.substr(0,sp)+"\x1f"+z.substr(sp+1)]=i;}
    }
}

std::vector<std::string> GgufTokenizer::pretokenize_gpt2(const std::string& text) const{
    // GPT-2/SmolLM compatible pre-tokenization. This mirrors the important semantic
    // boundaries of the canonical regex while remaining dependency-free: contractions,
    // optional leading ASCII space, letters, digits, punctuation, and whitespace runs.
    // Non-ASCII code points are treated as letters so their bytes remain in one BPE word.
    std::vector<std::string> out; std::size_t p=0;
    auto starts_contraction=[&](std::size_t at,std::size_t& n){
        static const char* cs[]={"'s","'t","'re","'ve","'m","'ll","'d"};
        for(auto*c:cs){std::size_t z=std::strlen(c);if(at+z<=text.size()&&text.compare(at,z,c)==0){n=z;return true;}}return false;
    };
    while(p<text.size()){
        std::size_t cn=0;if(starts_contraction(p,cn)){out.push_back(text.substr(p,cn));p+=cn;continue;}
        auto u=utf8_at(text,p);
        if(u_space(u.cp)){
            if(u.cp==0x20 && p+1<text.size()){
                auto n=utf8_at(text,p+1);
                if(!u_space(n.cp)){
                    std::size_t q=p+1; enum{L,D,P} kind=u_digit(n.cp)?D:(u_letter(n.cp)?L:P);
                    while(q<text.size()){auto x=utf8_at(text,q);auto k=u_digit(x.cp)?D:(u_letter(x.cp)?L:P);if(u_space(x.cp)||k!=kind)break;q+=x.len;}
                    out.push_back(text.substr(p,q-p));p=q;continue;
                }
            }
            std::size_t q=p+u.len;while(q<text.size()){auto x=utf8_at(text,q);if(!u_space(x.cp))break;q+=x.len;}out.push_back(text.substr(p,q-p));p=q;continue;
        }
        enum{L,D,P} kind=u_digit(u.cp)?D:(u_letter(u.cp)?L:P);std::size_t q=p+u.len;
        while(q<text.size()){std::size_t xcn=0;if(starts_contraction(q,xcn))break;auto x=utf8_at(text,q);auto k=u_digit(x.cp)?D:(u_letter(x.cp)?L:P);if(u_space(x.cp)||k!=kind)break;q+=x.len;}
        out.push_back(text.substr(p,q-p));p=q;
    }
    return out;
}

std::vector<int> GgufTokenizer::encode_gpt2(const std::string& text) const{
    std::vector<int> ids;
    if(!byte_bpe_complete_ && merge_rank_.empty()){
        // Compatibility path for small/non-standard GGUF fixtures that store literal
        // pieces instead of the canonical 256 GPT-2 byte alphabet.
        std::size_t p=0;while(p<text.size()){
            int best=-1;std::size_t bl=0;for(const auto&kv:greedy_){if(kv.first.empty()||kv.first.size()<=bl||kv.first.size()>text.size()-p)continue;if(text.compare(p,kv.first.size(),kv.first)==0){best=kv.second;bl=kv.first.size();}}
            if(best<0){if(unk_>=0){ids.push_back(unk_);++p;continue;}throw std::runtime_error("tokenizer could not encode byte sequence at offset "+std::to_string(p));}
            ids.push_back(best);p+=bl;
        }return ids;
    }
    auto bpe_one=[&](const std::string& raw){
        std::string s=gpt2_encode_bytes(raw);std::vector<std::string> syms;
        for(std::size_t i=0;i<s.size();){auto u=utf8_at(s,i);syms.push_back(s.substr(i,u.len));i+=u.len;}
        if(!merge_rank_.empty())for(;;){std::size_t br=std::numeric_limits<std::size_t>::max(),bp=0;bool found=false;for(std::size_t i=0;i+1<syms.size();++i){auto it=merge_rank_.find(syms[i]+"\x1f"+syms[i+1]);if(it!=merge_rank_.end()&&it->second<br){br=it->second;bp=i;found=true;}}if(!found)break;syms[bp]+=syms[bp+1];syms.erase(syms.begin()+static_cast<std::ptrdiff_t>(bp+1));}
        for(const auto&z:syms){auto it=token_to_id_.find(z);if(it!=token_to_id_.end()){ids.push_back(it->second);continue;}
            // A malformed/partial merge table must never make arbitrary input unencodable.
            // Decompose the symbol back to byte-unicode atoms and emit those pieces.
            for(std::size_t i=0;i<z.size();){auto u=utf8_at(z,i);auto atom=z.substr(i,u.len);auto jt=token_to_id_.find(atom);if(jt==token_to_id_.end()){if(unk_>=0)ids.push_back(unk_);else throw std::runtime_error("GPT-2 byte token absent from GGUF vocabulary");}else ids.push_back(jt->second);i+=u.len;}
        }
    };
    // Preserve GGUF control/user-defined tokens exactly when they occur verbatim.
    std::size_t p=0;while(p<text.size()){
        int special=-1;std::size_t sl=0;for(const auto&sp:specials_){const auto&t=sp.first;if(t.empty()||t.size()>text.size()-p)continue;if(text.compare(p,t.size(),t)==0){special=sp.second;sl=t.size();break;}}
        if(special>=0){ids.push_back(special);p+=sl;continue;}
        std::size_t q=p+1;while(q<text.size()){bool hit=false;for(const auto&sp:specials_){const auto&t=sp.first;if(!t.empty()&&t.size()<=text.size()-q&&text.compare(q,t.size(),t)==0){hit=true;break;}}if(hit)break;++q;}
        auto chunks=pretokenize_gpt2(text.substr(p,q-p));for(auto&c:chunks)bpe_one(c);p=q;
    }
    return ids;
}

std::vector<int> GgufTokenizer::encode_sentencepiece(const std::string& text) const{
    if(text.empty())return{};
    static const std::string marker="\xE2\x96\x81"; // U+2581 LOWER ONE EIGHTH BLOCK
    std::string s=replace_all(text," ",marker);if(add_space_prefix_)s=marker+s;
    struct Symbol{int prev{-1},next{-1};std::size_t off{},len{};};
    std::vector<Symbol> symbols;
    for(std::size_t off=0;off<s.size();){auto u=utf8_at(s,off);int idx=static_cast<int>(symbols.size());symbols.push_back({idx-1,off+u.len<s.size()?idx+1:-1,off,u.len});off+=u.len;}
    struct Bigram{int left{},right{};double score{};std::size_t size{};};
    struct Compare{bool operator()(const Bigram&l,const Bigram&r)const{return l.score<r.score||(l.score==r.score&&l.left>r.left);}};
    std::priority_queue<Bigram,std::vector<Bigram>,Compare> work;
    auto try_add=[&](int left,int right){if(left<0||right<0)return;const auto&L=symbols[static_cast<std::size_t>(left)];const auto&R=symbols[static_cast<std::size_t>(right)];if(!L.len||!R.len||L.next!=right||R.prev!=left)return;auto it=token_to_id_.find(s.substr(L.off,L.len+R.len));if(it==token_to_id_.end())return;int id=it->second;if(id<0||static_cast<std::size_t>(id)>=scores_.size())return;work.push({left,right,scores_[static_cast<std::size_t>(id)],L.len+R.len});};
    for(int i=1;i<static_cast<int>(symbols.size());++i)try_add(i-1,i);
    while(!work.empty()){
        auto b=work.top();work.pop();auto&L=symbols[static_cast<std::size_t>(b.left)];auto&R=symbols[static_cast<std::size_t>(b.right)];
        if(!L.len||!R.len||L.next!=b.right||R.prev!=b.left||L.len+R.len!=b.size)continue;
        L.len+=R.len;R.len=0;L.next=R.next;if(R.next>=0)symbols[static_cast<std::size_t>(R.next)].prev=b.left;
        try_add(L.prev,b.left);try_add(b.left,L.next);
    }
    std::vector<int> ids;
    for(int i=symbols.empty()?-1:0;i!=-1;i=symbols[static_cast<std::size_t>(i)].next){const auto&sym=symbols[static_cast<std::size_t>(i)];auto it=token_to_id_.find(s.substr(sym.off,sym.len));if(it!=token_to_id_.end()){ids.push_back(it->second);continue;}for(std::size_t j=0;j<sym.len;++j){unsigned char b=static_cast<unsigned char>(s[sym.off+j]);auto bt=byte_token_id_.find(b);if(bt!=byte_token_id_.end())ids.push_back(bt->second);else if(unk_>=0)ids.push_back(unk_);else throw std::runtime_error("SentencePiece byte fallback token missing");}}
    return ids;
}

std::vector<int> GgufTokenizer::encode(const std::string&text,bool add_bos)const{
    std::vector<int> ids;if(add_bos&&bos_>=0&&static_cast<std::size_t>(bos_)<tokens_.size())ids.push_back(bos_);
    std::vector<int> body;if(model_=="llama"||model_=="replit")body=encode_sentencepiece(text);else body=encode_gpt2(text);ids.insert(ids.end(),body.begin(),body.end());return ids;
}
std::string GgufTokenizer::decode_piece(int token)const{
    if(token<0||static_cast<std::size_t>(token)>=tokens_.size())return{};const auto&t=tokens_[token];
    if((model_=="llama"||model_=="replit")){
        if(static_cast<std::size_t>(token)<token_types_.size()&&token_types_[token]==6&&t.size()==6&&t.rfind("<0x",0)==0){try{int b=std::stoi(t.substr(3,2),nullptr,16);return std::string(1,static_cast<char>(b));}catch(...){return{};}}
        if(static_cast<std::size_t>(token)<token_types_.size()&&token_types_[token]==3)return t;
        return replace_all(t,"\xE2\x96\x81"," ");
    }
    return gpt2_decode_bytes(t);
}
std::string GgufTokenizer::decode(const std::vector<int>&ids)const{std::string s;for(int t:ids)s+=decode_piece(t);if((model_=="llama"||model_=="replit")&&add_space_prefix_&&!s.empty()&&s[0]==' ')s.erase(s.begin());return s;}

LlamaModel::LlamaModel(const std::string&p,unsigned th,std::size_t ctx,std::size_t page,KVCacheType kv_type):file_(p),tok_(file_),threads_(std::max(1u,th)),kv_type_(kv_type),pool_(std::max(1u,th)){load_config(ctx,page);}
void LlamaModel::load_config(std::size_t ctx_override,std::size_t kv_page_tokens){
    const auto arch=file_.get_string("general.architecture").value_or("");if(arch!="llama")throw std::runtime_error("MemVanta real-model path currently supports GGUF architecture=llama, got: "+arch);
    auto req=[&](const char*k){auto v=file_.get_u64(k);if(!v)throw std::runtime_error(std::string("missing GGUF metadata: ")+k);return static_cast<std::size_t>(*v);};
    c_.n_ctx=req("llama.context_length");c_.n_embd=req("llama.embedding_length");c_.n_ff=req("llama.feed_forward_length");c_.n_layer=req("llama.block_count");c_.n_head=req("llama.attention.head_count");c_.n_head_kv=static_cast<std::size_t>(file_.get_u64("llama.attention.head_count_kv").value_or(c_.n_head));c_.rms_eps=static_cast<float>(file_.get_f64("llama.attention.layer_norm_rms_epsilon").value_or(1e-5));c_.rope_theta=static_cast<float>(file_.get_f64("llama.rope.freq_base").value_or(10000.0));
    if(!c_.n_ctx||!c_.n_embd||!c_.n_ff||!c_.n_layer||!c_.n_head||!c_.n_head_kv)throw std::runtime_error("invalid zero Llama dimension in GGUF metadata");
    if(c_.n_embd%c_.n_head||c_.n_head%c_.n_head_kv)throw std::runtime_error("invalid Llama head configuration");
    if(c_.head_dim()%2)throw std::runtime_error("Llama head dimension must be even for RoPE");
    if(!std::isfinite(c_.rms_eps)||c_.rms_eps<=0||!std::isfinite(c_.rope_theta)||c_.rope_theta<=0)throw std::runtime_error("invalid Llama numeric metadata");
    auto expect_matrix=[&](const GgufTensor&t,std::size_t cols,std::size_t rows){if(t.dims.size()!=2||t.ne(0)!=cols||t.ne(1)!=rows)throw std::runtime_error("tensor shape mismatch: "+t.name+" expected ["+std::to_string(cols)+","+std::to_string(rows)+"]");};
    token_embd_=&file_.tensor("token_embd.weight");if(token_embd_->dims.size()!=2||token_embd_->ne(0)!=c_.n_embd)throw std::runtime_error("token embedding shape mismatch");c_.vocab=static_cast<std::size_t>(token_embd_->ne(1));if(!c_.vocab)throw std::runtime_error("token embedding vocabulary is empty");if(ctx_override)c_.n_ctx=std::min(c_.n_ctx,ctx_override);if(!c_.n_ctx)throw std::runtime_error("context length must be > 0");
    output_norm_=&file_.tensor("output_norm.weight");output_=file_.has_tensor("output.weight")?&file_.tensor("output.weight"):token_embd_;expect_matrix(*output_,c_.n_embd,c_.vocab);
    output_norm_f_.resize(c_.n_embd);tensor_vector_to_f32(file_,*output_norm_,output_norm_f_.data(),c_.n_embd);
    layers_.resize(c_.n_layer);kv_.reserve(c_.n_layer);const std::size_t K=c_.kv_dim();
    for(std::size_t i=0;i<c_.n_layer;++i){auto&L=layers_[i];auto n=[&](const char*s){return "blk."+std::to_string(i)+s;};L.attn_norm=&file_.tensor(n(".attn_norm.weight"));L.q=&file_.tensor(n(".attn_q.weight"));L.k=&file_.tensor(n(".attn_k.weight"));L.v=&file_.tensor(n(".attn_v.weight"));L.o=&file_.tensor(n(".attn_output.weight"));L.ffn_norm=&file_.tensor(n(".ffn_norm.weight"));L.gate=&file_.tensor(n(".ffn_gate.weight"));L.up=&file_.tensor(n(".ffn_up.weight"));L.down=&file_.tensor(n(".ffn_down.weight"));expect_matrix(*L.q,c_.n_embd,c_.n_embd);expect_matrix(*L.k,c_.n_embd,K);expect_matrix(*L.v,c_.n_embd,K);expect_matrix(*L.o,c_.n_embd,c_.n_embd);expect_matrix(*L.gate,c_.n_embd,c_.n_ff);expect_matrix(*L.up,c_.n_embd,c_.n_ff);expect_matrix(*L.down,c_.n_ff,c_.n_embd);L.attn_norm_f.resize(c_.n_embd);L.ffn_norm_f.resize(c_.n_embd);tensor_vector_to_f32(file_,*L.attn_norm,L.attn_norm_f.data(),c_.n_embd);tensor_vector_to_f32(file_,*L.ffn_norm,L.ffn_norm_f.data(),c_.n_embd);kv_.emplace_back(K,kv_page_tokens,kv_type_);}
    x_.resize(c_.n_embd);n_.resize(c_.n_embd);q_.resize(c_.n_embd);k_.resize(K);v_.resize(K);att_.resize(c_.n_embd);proj_.resize(c_.n_embd);gate_.resize(c_.n_ff);up_.resize(c_.n_ff);ff_.resize(c_.n_ff);out_.resize(c_.n_embd);logits_.resize(c_.vocab);scores_.reserve(c_.n_ctx);
    const std::size_t half=c_.head_dim()/2;rope_cos_.resize(c_.n_ctx*half);rope_sin_.resize(c_.n_ctx*half);for(std::size_t p=0;p<c_.n_ctx;++p)for(std::size_t i=0;i<half;++i){float inv=std::pow(c_.rope_theta,-2.0f*float(i)/float(c_.head_dim()));float a=float(p)*inv;rope_cos_[p*half+i]=std::cos(a);rope_sin_[p*half+i]=std::sin(a);}
    reset();
}
void LlamaModel::reset(){pos_=0;for(auto&k:kv_)k.reset();}
void LlamaModel::rmsnorm(const std::vector<float>&x,const std::vector<float>&w,std::vector<float>&o)const{double ss=0;for(float v:x)ss+=static_cast<double>(v)*v;float inv=1.0f/std::sqrt(static_cast<float>(ss/x.size())+c_.rms_eps);for(std::size_t i=0;i<x.size();++i)o[i]=x[i]*inv*w[i];}
void LlamaModel::rope_ptr(float*q,float*k,std::size_t pos)const{const std::size_t hd=c_.head_dim(),half=hd/2;const float*co=rope_cos_.data()+pos*half,*si=rope_sin_.data()+pos*half;auto rot=[&](float*z,std::size_t heads){for(std::size_t h=0;h<heads;++h){float*b=z+h*hd;for(std::size_t i=0;i<half;++i){float x0=b[i],x1=b[i+half],c=co[i],s=si[i];b[i]=x0*c-x1*s;b[i+half]=x0*s+x1*c;}}};rot(q,c_.n_head);rot(k,c_.n_head_kv);}
void LlamaModel::rope(std::vector<float>&q,std::vector<float>&k,std::size_t pos)const{rope_ptr(q.data(),k.data(),pos);}
const std::vector<float>& LlamaModel::forward_batch_impl(const int*tokens,std::size_t B,bool compute_logits_last){
    if(!B)return logits_;if(pos_+B>c_.n_ctx)throw std::runtime_error("context exhausted");for(std::size_t b=0;b<B;++b)if(tokens[b]<0||static_cast<std::size_t>(tokens[b])>=c_.vocab)throw std::runtime_error("token id out of range");
    const std::size_t D=c_.n_embd,F=c_.n_ff,K=c_.kv_dim(),hd=c_.head_dim(),group=c_.n_head/c_.n_head_kv,start=pos_;
    std::vector<float>X(B*D),N(B*D),Q(B*D),KK(B*K),VV(B*K),ATT(B*D),PROJ(B*D),G(B*F),U(B*F),FF(B*F),OUT(B*D);
    for(std::size_t b=0;b<B;++b)tensor_read_row_f32(file_,*token_embd_,tokens[b],X.data()+b*D,D);
    auto rms_batch=[&](const std::vector<float>&in,const std::vector<float>&w,std::vector<float>&out){pool_.parallel_for(B,[&](std::size_t a,std::size_t z){for(std::size_t b=a;b<z;++b){const float*x=in.data()+b*D;float*o=out.data()+b*D;double ss=0;for(std::size_t i=0;i<D;++i)ss+=double(x[i])*x[i];float inv=1.0f/std::sqrt(float(ss/D)+c_.rms_eps);for(std::size_t i=0;i<D;++i)o[i]=x[i]*inv*w[i];}});};
    for(std::size_t li=0;li<c_.n_layer;++li){auto&L=layers_[li];rms_batch(X,L.attn_norm_f,N);tensor_matmul_batch_v06(file_,*L.q,N.data(),Q.data(),B,threads_,&pool_);tensor_matmul_batch_v06(file_,*L.k,N.data(),KK.data(),B,threads_,&pool_);tensor_matmul_batch_v06(file_,*L.v,N.data(),VV.data(),B,threads_,&pool_);
        for(std::size_t b=0;b<B;++b){rope_ptr(Q.data()+b*D,KK.data()+b*K,start+b);kv_[li].write(start+b,KK.data()+b*K,VV.data()+b*K);}
        std::fill(ATT.begin(),ATT.end(),0.0f);pool_.parallel_for(B*c_.n_head,[&](std::size_t a,std::size_t z){std::vector<float> scores;for(std::size_t job=a;job<z;++job){std::size_t b=job/c_.n_head,h=job%c_.n_head,kh=h/group,qb=h*hd,kb=kh*hd,upto=start+b;scores.resize(upto+1);float mx=-std::numeric_limits<float>::infinity();for(std::size_t t=0;t<=upto;++t){float sf=kv_[li].dot_key(t,kb,Q.data()+b*D+qb,hd)/std::sqrt(float(hd));scores[t]=sf;mx=std::max(mx,sf);}double den=0;for(float&v:scores){v=std::exp(v-mx);den+=v;}float inv=float(1.0/den);float*out=ATT.data()+b*D+qb;for(std::size_t t=0;t<=upto;++t)kv_[li].add_value(t,kb,scores[t]*inv,out,hd);}});
        tensor_matmul_batch_v06(file_,*L.o,ATT.data(),PROJ.data(),B,threads_,&pool_);pool_.parallel_for(B*D,[&](std::size_t a,std::size_t z){for(std::size_t i=a;i<z;++i)X[i]+=PROJ[i];});rms_batch(X,L.ffn_norm_f,N);tensor_matmul_batch_v06(file_,*L.gate,N.data(),G.data(),B,threads_,&pool_);tensor_matmul_batch_v06(file_,*L.up,N.data(),U.data(),B,threads_,&pool_);pool_.parallel_for(B*F,[&](std::size_t a,std::size_t z){for(std::size_t i=a;i<z;++i){float x=G[i];FF[i]=(x/(1.0f+std::exp(-x)))*U[i];}});tensor_matmul_batch_v06(file_,*L.down,FF.data(),OUT.data(),B,threads_,&pool_);pool_.parallel_for(B*D,[&](std::size_t a,std::size_t z){for(std::size_t i=a;i<z;++i)X[i]+=OUT[i];});
    }
    pos_+=B;if(compute_logits_last){const float*x=X.data()+(B-1)*D;double ss=0;for(std::size_t i=0;i<D;++i)ss+=double(x[i])*x[i];float inv=1.0f/std::sqrt(float(ss/D)+c_.rms_eps);for(std::size_t i=0;i<D;++i)n_[i]=x[i]*inv*output_norm_f_[i];tensor_matvec(file_,*output_,n_.data(),logits_.data(),threads_,&pool_);}return logits_;
}
const std::vector<float>& LlamaModel::forward(int token,bool compute_logits){
    if(token<0||static_cast<std::size_t>(token)>=c_.vocab)throw std::runtime_error("token id out of range");if(pos_>=c_.n_ctx)throw std::runtime_error("context exhausted");tensor_read_row_f32(file_,*token_embd_,static_cast<std::size_t>(token),x_.data(),c_.n_embd);const std::size_t hd=c_.head_dim(),group=c_.n_head/c_.n_head_kv;
    for(std::size_t li=0;li<c_.n_layer;++li){auto&L=layers_[li];rmsnorm(x_,L.attn_norm_f,n_);tensor_matvec(file_,*L.q,n_.data(),q_.data(),threads_,nullptr);tensor_matvec(file_,*L.k,n_.data(),k_.data(),threads_,nullptr);tensor_matvec(file_,*L.v,n_.data(),v_.data(),threads_,nullptr);rope(q_,k_,pos_);kv_[li].write(pos_,k_.data(),v_.data());std::fill(att_.begin(),att_.end(),0.0f);
        scores_.resize(pos_+1);for(std::size_t h=0;h<c_.n_head;++h){std::size_t kh=h/group,qb=h*hd,kb=kh*hd;float mx=-std::numeric_limits<float>::infinity();for(std::size_t t=0;t<=pos_;++t){float sf=kv_[li].dot_key(t,kb,q_.data()+qb,hd)/std::sqrt(float(hd));scores_[t]=sf;mx=std::max(mx,sf);}double den=0;for(std::size_t t=0;t<=pos_;++t){scores_[t]=std::exp(scores_[t]-mx);den+=scores_[t];}float inv=float(1.0/den);for(std::size_t t=0;t<=pos_;++t)kv_[li].add_value(t,kb,scores_[t]*inv,att_.data()+qb,hd);}
        tensor_matvec(file_,*L.o,att_.data(),proj_.data(),threads_,nullptr);for(std::size_t i=0;i<c_.n_embd;++i)x_[i]+=proj_[i];rmsnorm(x_,L.ffn_norm_f,n_);tensor_matvec(file_,*L.gate,n_.data(),gate_.data(),threads_,nullptr);tensor_matvec(file_,*L.up,n_.data(),up_.data(),threads_,nullptr);for(std::size_t i=0;i<c_.n_ff;++i){float z=gate_[i];ff_[i]=(z/(1.0f+std::exp(-z)))*up_[i];}tensor_matvec(file_,*L.down,ff_.data(),out_.data(),threads_,nullptr);for(std::size_t i=0;i<c_.n_embd;++i)x_[i]+=out_[i];}
    ++pos_;if(compute_logits){rmsnorm(x_,output_norm_f_,n_);tensor_matvec(file_,*output_,n_.data(),logits_.data(),threads_,nullptr);}return logits_;
}
const std::vector<float>& LlamaModel::prefill(const std::vector<int>&tokens,std::size_t batch_size,bool compute_logits_last){if(tokens.empty())return logits_;batch_size=std::max<std::size_t>(1,batch_size);for(std::size_t i=0;i<tokens.size();i+=batch_size){auto n=std::min(batch_size,tokens.size()-i);forward_batch_impl(tokens.data()+i,n,compute_logits_last&&i+n==tokens.size());}return logits_;}
std::vector<int> LlamaModel::generate(const std::vector<int>&prompt,std::size_t n,Sampler&sampler){reset();if(prompt.empty())throw std::runtime_error("prompt cannot be empty");if(prompt.size()>1){std::vector<int>pre(prompt.begin(),prompt.end()-1);prefill(pre,64,false);}int cur=prompt.back();std::vector<int>out;out.reserve(n);for(std::size_t i=0;i<n;++i){const auto&l=forward(cur,true);int next=sampler.sample(l);out.push_back(next);cur=next;if(next==tok_.eos_id())break;}return out;}

std::size_t LlamaModel::kv_bytes_allocated()const{std::size_t n=0;for(auto&k:kv_)n+=k.bytes_allocated();return n;}

} // namespace memvanta