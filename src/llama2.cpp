#include "memvanta/llama2.hpp"

#include "memvanta/matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace memvanta {
namespace {
using Clock = std::chrono::steady_clock;

std::size_t peak_rss_kib() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<std::size_t>(ru.ru_maxrss);
}

inline void rmsnorm(float* out, const float* x, const float* weight, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i)
        ss += static_cast<double>(x[i]) * x[i];
    const float scale = 1.0f / std::sqrt(static_cast<float>(ss / n) + 1e-5f);
    for (int i = 0; i < n; ++i)
        out[i] = weight[i] * x[i] * scale;
}

inline void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        mx = std::max(mx, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}
} // namespace

struct Llama2Model::Weights {
    const float* token_embedding{};
    const float* rms_att{};
    const float* wq{};
    const float* wk{};
    const float* wv{};
    const float* wo{};
    const float* rms_ffn{};
    const float* w1{};
    const float* w2{};
    const float* w3{};
    const float* rms_final{};
    const float* classifier{};
};

struct Llama2Model::State {
    std::vector<float> x, xb, xb2, hb, hb2, q;
    std::vector<float> key_cache, value_cache, att, logits;
};

Llama2Model::Llama2Model(const std::string& path, unsigned threads)
    : threads_(threads ? threads : 1) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("cannot open checkpoint: " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fstat failed");
    }
    file_size_ = static_cast<std::size_t>(st.st_size);
    if (file_size_ < sizeof(Llama2Config))
        throw std::runtime_error("checkpoint too small");
    map_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
        throw std::runtime_error("mmap failed");
    }
    std::memcpy(&cfg_, map_, sizeof(cfg_));
    shared_classifier_ = cfg_.vocab_size > 0;
    if (cfg_.vocab_size < 0)
        cfg_.vocab_size = -cfg_.vocab_size;
    if (cfg_.dim <= 0 || cfg_.hidden_dim <= 0 || cfg_.n_layers <= 0 || cfg_.n_heads <= 0 ||
        cfg_.n_kv_heads <= 0 || cfg_.vocab_size <= 0 || cfg_.seq_len <= 0)
        throw std::runtime_error("invalid checkpoint config");
    if (cfg_.dim % cfg_.n_heads || cfg_.n_heads % cfg_.n_kv_heads)
        throw std::runtime_error("invalid head configuration");
    map_weights();

    const int kv_dim = cfg_.dim * cfg_.n_kv_heads / cfg_.n_heads;
    s_ = new State;
    s_->x.resize(cfg_.dim);
    s_->xb.resize(cfg_.dim);
    s_->xb2.resize(cfg_.dim);
    s_->hb.resize(cfg_.hidden_dim);
    s_->hb2.resize(cfg_.hidden_dim);
    s_->q.resize(cfg_.dim);
    s_->key_cache.resize(static_cast<std::size_t>(cfg_.n_layers) * cfg_.seq_len * kv_dim);
    s_->value_cache.resize(static_cast<std::size_t>(cfg_.n_layers) * cfg_.seq_len * kv_dim);
    s_->att.resize(static_cast<std::size_t>(cfg_.n_heads) * cfg_.seq_len);
    s_->logits.resize(cfg_.vocab_size);
}

Llama2Model::~Llama2Model() {
    delete s_;
    delete w_;
    if (map_)
        munmap(map_, file_size_);
    if (fd_ >= 0)
        ::close(fd_);
}

void Llama2Model::map_weights() {
    w_ = new Weights;
    const auto* begin = reinterpret_cast<const std::byte*>(map_);
    const float* p = reinterpret_cast<const float*>(begin + sizeof(Llama2Config));
    const float* end = reinterpret_cast<const float*>(begin + file_size_);
    auto take = [&](std::size_t n) -> const float* {
        if (n > static_cast<std::size_t>(end - p))
            throw std::runtime_error("checkpoint truncated");
        const float* out = p;
        p += n;
        return out;
    };
    const std::size_t d = cfg_.dim, h = cfg_.hidden_dim, L = cfg_.n_layers, V = cfg_.vocab_size;
    const std::size_t head = d / cfg_.n_heads;
    const std::size_t kv = cfg_.n_kv_heads * head;
    w_->token_embedding = take(V * d);
    w_->rms_att = take(L * d);
    w_->wq = take(L * d * d);
    w_->wk = take(L * d * kv);
    w_->wv = take(L * d * kv);
    w_->wo = take(L * d * d);
    w_->rms_ffn = take(L * d);
    w_->w1 = take(L * d * h);
    w_->w2 = take(L * h * d);
    w_->w3 = take(L * d * h);
    w_->rms_final = take(d);
    // Legacy checkpoints carry two RoPE tables. Modern runtimes recompute RoPE,
    // so retain format compatibility while avoiding those reads during inference.
    take(static_cast<std::size_t>(cfg_.seq_len) * head / 2);
    take(static_cast<std::size_t>(cfg_.seq_len) * head / 2);
    w_->classifier = shared_classifier_ ? w_->token_embedding : take(V * d);
}

void Llama2Model::clear_kv() {
    std::fill(s_->key_cache.begin(), s_->key_cache.end(), 0.0f);
    std::fill(s_->value_cache.begin(), s_->value_cache.end(), 0.0f);
}

const float* Llama2Model::forward(int token, int pos) {
    if (token < 0 || token >= cfg_.vocab_size)
        throw std::runtime_error("token out of range");
    if (pos < 0 || pos >= cfg_.seq_len)
        throw std::runtime_error("position out of range");
    const int d = cfg_.dim, hd = cfg_.hidden_dim, head = d / cfg_.n_heads;
    const int kv_dim = d * cfg_.n_kv_heads / cfg_.n_heads;
    const int kv_mul = cfg_.n_heads / cfg_.n_kv_heads;
    auto& S = *s_;
    auto& W = *w_;
    std::memcpy(
        S.x.data(), W.token_embedding + static_cast<std::size_t>(token) * d, d * sizeof(float));

    for (int l = 0; l < cfg_.n_layers; ++l) {
        rmsnorm(S.xb.data(), S.x.data(), W.rms_att + static_cast<std::size_t>(l) * d, d);
        float* k = S.key_cache.data() + (static_cast<std::size_t>(l) * cfg_.seq_len + pos) * kv_dim;
        float* v =
            S.value_cache.data() + (static_cast<std::size_t>(l) * cfg_.seq_len + pos) * kv_dim;
        matvec_f32(
            W.wq + static_cast<std::size_t>(l) * d * d, S.xb.data(), S.q.data(), d, d, threads_);
        matvec_f32(
            W.wk + static_cast<std::size_t>(l) * d * kv_dim, S.xb.data(), k, kv_dim, d, threads_);
        matvec_f32(
            W.wv + static_cast<std::size_t>(l) * d * kv_dim, S.xb.data(), v, kv_dim, d, threads_);

        for (int i = 0; i < d; i += 2) {
            const int head_dim = i % head;
            const float freq = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / head);
            const float ang = pos * freq, c = std::cos(ang), sn = std::sin(ang);
            float a = S.q[i], b = S.q[i + 1];
            S.q[i] = a * c - b * sn;
            S.q[i + 1] = a * sn + b * c;
            if (i < kv_dim) {
                a = k[i];
                b = k[i + 1];
                k[i] = a * c - b * sn;
                k[i + 1] = a * sn + b * c;
            }
        }

        for (int hidx = 0; hidx < cfg_.n_heads; ++hidx) {
            float* at = S.att.data() + static_cast<std::size_t>(hidx) * cfg_.seq_len;
            const float* q = S.q.data() + hidx * head;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = S.key_cache.data() +
                                  (static_cast<std::size_t>(l) * cfg_.seq_len + t) * kv_dim +
                                  (hidx / kv_mul) * head;
                float sc = 0.0f;
                for (int i = 0; i < head; ++i)
                    sc += q[i] * kt[i];
                at[t] = sc / std::sqrt(static_cast<float>(head));
            }
            softmax(at, pos + 1);
            float* xb = S.xb.data() + hidx * head;
            std::fill(xb, xb + head, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = S.value_cache.data() +
                                  (static_cast<std::size_t>(l) * cfg_.seq_len + t) * kv_dim +
                                  (hidx / kv_mul) * head;
                const float a = at[t];
                for (int i = 0; i < head; ++i)
                    xb[i] += a * vt[i];
            }
        }
        matvec_f32(
            W.wo + static_cast<std::size_t>(l) * d * d, S.xb.data(), S.xb2.data(), d, d, threads_);
        for (int i = 0; i < d; ++i)
            S.x[i] += S.xb2[i];

        rmsnorm(S.xb.data(), S.x.data(), W.rms_ffn + static_cast<std::size_t>(l) * d, d);
        matvec_f32(
            W.w1 + static_cast<std::size_t>(l) * d * hd, S.xb.data(), S.hb.data(), hd, d, threads_);
        matvec_f32(W.w3 + static_cast<std::size_t>(l) * d * hd,
                   S.xb.data(),
                   S.hb2.data(),
                   hd,
                   d,
                   threads_);
        for (int i = 0; i < hd; ++i) {
            const float z = S.hb[i];
            S.hb[i] = (z / (1.0f + std::exp(-z))) * S.hb2[i];
        }
        matvec_f32(
            W.w2 + static_cast<std::size_t>(l) * hd * d, S.hb.data(), S.xb.data(), d, hd, threads_);
        for (int i = 0; i < d; ++i)
            S.x[i] += S.xb[i];
    }
    rmsnorm(S.x.data(), S.x.data(), W.rms_final, d);
    matvec_f32(W.classifier, S.x.data(), S.logits.data(), cfg_.vocab_size, d, threads_);
    return S.logits.data();
}

int Llama2Model::greedy(int token, int pos) {
    const float* lg = forward(token, pos);
    return static_cast<int>(std::max_element(lg, lg + cfg_.vocab_size) - lg);
}

Llama2BenchResult
Llama2Model::benchmark(std::size_t prompt_tokens, std::size_t generated_tokens, unsigned warmup) {
    if (prompt_tokens > static_cast<std::size_t>(cfg_.seq_len))
        throw std::runtime_error("prompt exceeds model context");
    if (generated_tokens > static_cast<std::size_t>(cfg_.seq_len))
        throw std::runtime_error("generation exceeds model context");
    auto token_at = [&](std::size_t i) {
        return static_cast<int>((i * 37 + 11) % cfg_.vocab_size);
    };
    for (unsigned w = 0; w < warmup; ++w) {
        clear_kv();
        const std::size_t n = std::min<std::size_t>(8, prompt_tokens);
        for (std::size_t i = 0; i < n; ++i)
            forward(token_at(i), static_cast<int>(i));
    }

    clear_kv();
    auto p0 = Clock::now();
    for (std::size_t i = 0; i < prompt_tokens; ++i)
        forward(token_at(i), static_cast<int>(i));
    auto p1 = Clock::now();
    const double psec = std::chrono::duration<double>(p1 - p0).count();

    clear_kv();
    int tok = 1 % cfg_.vocab_size;
    auto g0 = Clock::now();
    for (std::size_t i = 0; i < generated_tokens; ++i)
        tok = greedy(tok, static_cast<int>(i));
    auto g1 = Clock::now();
    const double gsec = std::chrono::duration<double>(g1 - g0).count();

    clear_kv();
    auto c0 = Clock::now();
    const std::size_t client_prompt = std::min<std::size_t>(128, cfg_.seq_len - 1);
    for (std::size_t i = 0; i < client_prompt; ++i)
        forward(token_at(i), static_cast<int>(i));
    auto c1 = Clock::now();
    int t = greedy(1 % cfg_.vocab_size, static_cast<int>(client_prompt));
    auto c2 = Clock::now();
    const std::size_t remain = std::min<std::size_t>(
        generated_tokens > 1 ? generated_tokens - 1 : 0, cfg_.seq_len - client_prompt - 1);
    for (std::size_t i = 0; i < remain; ++i)
        t = greedy(t, static_cast<int>(client_prompt + 1 + i));
    auto c3 = Clock::now();
    (void)c1;

    Llama2BenchResult r;
    r.prompt_tps = prompt_tokens / psec;
    r.generation_tps = generated_tokens / gsec;
    r.ttft_ms = std::chrono::duration<double, std::milli>(c2 - c0).count();
    const double osec = std::chrono::duration<double>(c3 - c2).count();
    r.output_tps = remain ? remain / osec : 0.0;
    r.peak_rss_kib = peak_rss_kib();
    return r;
}

} // namespace memvanta
