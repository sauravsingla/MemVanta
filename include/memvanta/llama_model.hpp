#pragma once
#include "memvanta/gguf.hpp"
#include "memvanta/worker_pool.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace memvanta {

struct LlamaConfig {
    std::size_t n_ctx{};
    std::size_t n_embd{};
    std::size_t n_ff{};
    std::size_t n_layer{};
    std::size_t n_head{};
    std::size_t n_head_kv{};
    std::size_t vocab{};
    float rms_eps{1e-5f};
    float rope_theta{10000.0f};
    std::size_t head_dim() const { return n_embd/n_head; }
    std::size_t kv_dim() const { return n_head_kv*head_dim(); }
};

enum class KVCacheType { F32, F16, Q8 };
KVCacheType parse_kv_cache_type(const std::string& s);
const char* kv_cache_type_name(KVCacheType t);

class PagedKVCache {
public:
    explicit PagedKVCache(std::size_t kv_dim=0,std::size_t page_tokens=128,KVCacheType type=KVCacheType::F32);
    void reset();
    void write(std::size_t pos,const float*k,const float*v);
    float dot_key(std::size_t pos,std::size_t offset,const float*q,std::size_t n) const;
    void add_value(std::size_t pos,std::size_t offset,float alpha,float*out,std::size_t n) const;
    std::size_t bytes_allocated() const;
    KVCacheType type() const { return type_; }
private:
    struct Page {
        std::vector<float> k32,v32;
        std::vector<std::uint16_t> k16,v16;
        std::vector<std::int8_t> k8,v8;
        std::vector<float> ks,vs;
        Page(std::size_t elems,std::size_t tokens,KVCacheType type);
    };
    std::size_t kv_dim_{}, page_tokens_{}; KVCacheType type_{KVCacheType::F32};
    std::vector<std::unique_ptr<Page>> pages_;
    Page& ensure(std::size_t p);
};

struct SamplerConfig {
    float temperature{0.0f}; // 0 = greedy
    std::size_t top_k{40};
    std::uint64_t seed{1234};
};

class Sampler {
public:
    explicit Sampler(SamplerConfig c={});
    int sample(const std::vector<float>& logits);
private:
    SamplerConfig c_; std::mt19937_64 rng_;
};

class GgufTokenizer {
public:
    explicit GgufTokenizer(const GgufFile& file);
    std::vector<int> encode(const std::string& text,bool add_bos=false) const;
    std::string decode_piece(int token) const;
    std::string decode(const std::vector<int>& ids) const;
    std::size_t vocab_size() const { return tokens_.size(); }
    int bos_id() const { return bos_; }
    int eos_id() const { return eos_; }
    int unk_id() const { return unk_; }
    const std::string& model_type() const { return model_; }
    const std::string& pre_type() const { return pre_; }
private:
    std::vector<std::string> tokens_;
    std::vector<double> scores_;
    std::vector<std::int64_t> token_types_;
    int bos_{-1},eos_{-1},unk_{-1};
    bool add_space_prefix_{false};
    bool byte_bpe_complete_{false};
    std::string model_{"gpt2"},pre_{"default"};
    std::vector<std::pair<std::string,int>> greedy_;
    std::vector<std::pair<std::string,int>> specials_;
    std::unordered_map<std::string,int> token_to_id_;
    std::unordered_map<std::string,std::size_t> merge_rank_;
    std::unordered_map<unsigned char,int> byte_token_id_;
    static std::string gpt2_encode_bytes(const std::string& s);
    static std::string gpt2_decode_bytes(const std::string& s);
    std::vector<int> encode_gpt2(const std::string& text) const;
    std::vector<int> encode_sentencepiece(const std::string& text) const;
    std::vector<std::string> pretokenize_gpt2(const std::string& text) const;
};

class LlamaModel {
public:
    LlamaModel(const std::string& gguf_path,unsigned threads=1,std::size_t ctx_override=0,std::size_t kv_page_tokens=128,KVCacheType kv_type=KVCacheType::F16);
    const LlamaConfig& config() const { return c_; }
    const GgufFile& gguf() const { return file_; }
    const GgufTokenizer& tokenizer() const { return tok_; }
    void reset();
    const std::vector<float>& forward(int token,bool compute_logits=true);
    const std::vector<float>& prefill(const std::vector<int>& tokens,std::size_t batch_size=64,bool compute_logits_last=true);
    std::vector<int> generate(const std::vector<int>& prompt,std::size_t n,Sampler& sampler);
    std::size_t position() const { return pos_; }
    std::size_t kv_bytes_allocated() const;
    std::size_t rope_bytes_allocated() const;
    std::size_t scratch_bytes_allocated() const;
    unsigned threads() const { return threads_; }
    KVCacheType kv_type() const { return kv_type_; }
private:
    struct LayerWeights {
        const GgufTensor *attn_norm{},*q{},*k{},*v{},*o{},*ffn_norm{},*gate{},*up{},*down{};
        std::vector<float> attn_norm_f,ffn_norm_f;
    };
    GgufFile file_;
    GgufTokenizer tok_;
    LlamaConfig c_;
    unsigned threads_{};
    KVCacheType kv_type_{KVCacheType::F16};
    WorkerPool pool_;
    const GgufTensor *token_embd_{},*output_{},*output_norm_{};
    std::vector<float> output_norm_f_;
    std::vector<LayerWeights> layers_;
    std::vector<PagedKVCache> kv_;
    std::size_t pos_{};

    std::vector<float> x_,n_,q_,k_,v_,att_,proj_,gate_,up_,ff_,out_,logits_,scores_;
    // Batched prefill scratch is retained and reused instead of allocating eleven
    // large vectors for every prefill chunk.
    std::vector<float> batch_x_,batch_n_,batch_q_,batch_k_,batch_v_,batch_att_,batch_proj_,batch_gate_,batch_up_,batch_ff_,batch_out_;
    // RoPE tables grow only to the highest position actually used, not n_ctx.
    std::vector<float> rope_cos_,rope_sin_;
    std::size_t rope_cached_positions_{};
    void load_config(std::size_t ctx_override,std::size_t kv_page_tokens);
    const std::vector<float>& forward_batch_impl(const int* tokens,std::size_t batch,bool compute_logits_last);
    void rmsnorm(const std::vector<float>&x,const std::vector<float>&w,std::vector<float>&out) const;
    void ensure_rope(std::size_t last_pos);
    void rope(std::vector<float>&q,std::vector<float>&k,std::size_t pos) const;
    void rope_ptr(float* q,float* k,std::size_t pos) const;
};

} // namespace memvanta
