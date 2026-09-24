#include "memvanta/quant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
struct Config {
    std::size_t hidden = 768;
    std::size_t ff = 2048;
    std::size_t layers = 8;
    std::size_t heads = 12;
    std::size_t vocab = 4096;
    std::size_t max_ctx = 1024;
    unsigned threads = 5;
    unsigned reps = 5;
    unsigned warmup = 1;
};

struct Stat {
    double mean{}, sd{}, min{}, max{};
};
Stat calc(const std::vector<double>& v) {
    Stat s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double q = 0.0;
    for (double x : v)
        q += (x - s.mean) * (x - s.mean);
    s.sd = v.size() > 1 ? std::sqrt(q / (v.size() - 1)) : 0.0;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    return s;
}

std::size_t peak_rss_kib() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<std::size_t>(ru.ru_maxrss);
}

struct Q8Matrix {
    std::size_t rows{}, cols{};
    std::vector<memvanta::BlockQ8_0> blocks;

    Q8Matrix() = default;
    Q8Matrix(std::size_t r, std::size_t c, std::uint64_t seed) : rows(r), cols(c) {
        if (c % memvanta::QK)
            throw std::runtime_error("Q8 matrix cols must be multiple of 32");
        blocks.resize(r * c / memvanta::QK);
        // Direct deterministic quantized initialization: avoids multi-GB temporary FP32 weights.
        std::uint64_t x = seed ? seed : 1;
        auto rnd = [&]() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            return x;
        };
        const float base = 0.018f / std::sqrt(static_cast<float>(c));
        for (auto& b : blocks) {
            b.d = base;
            for (std::size_t i = 0; i < memvanta::QK; ++i) {
                int v = static_cast<int>(rnd() % 15) - 7;
                b.qs[i] = static_cast<std::int8_t>(v);
            }
        }
    }
    void mul(const float* x, float* y, unsigned threads) const {
        memvanta::matvec_q8_0(blocks.data(), x, y, rows, cols, threads);
    }
    std::uint64_t params() const { return static_cast<std::uint64_t>(rows) * cols; }
    std::uint64_t bytes() const {
        return static_cast<std::uint64_t>(blocks.size()) * sizeof(memvanta::BlockQ8_0);
    }
};

struct Layer {
    Q8Matrix q, k, v, o, gate, up, down;
    Layer(std::size_t h, std::size_t ff, std::uint64_t seed)
        : q(h, h, seed + 1), k(h, h, seed + 2), v(h, h, seed + 3), o(h, h, seed + 4),
          gate(ff, h, seed + 5), up(ff, h, seed + 6), down(h, ff, seed + 7) {}
    std::uint64_t params() const {
        return q.params() + k.params() + v.params() + o.params() + gate.params() + up.params() +
               down.params();
    }
    std::uint64_t bytes() const {
        return q.bytes() + k.bytes() + v.bytes() + o.bytes() + gate.bytes() + up.bytes() +
               down.bytes();
    }
};

inline void rmsnorm(std::vector<float>& out, const std::vector<float>& x) {
    double ss = 0.0;
    for (float v : x)
        ss += static_cast<double>(v) * v;
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / x.size()) + 1e-5f);
    out.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        out[i] = x[i] * inv;
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void rope(std::vector<float>& q, std::vector<float>& k, std::size_t pos, std::size_t heads) {
    const std::size_t h = q.size(), hd = h / heads;
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t b = head * hd;
        for (std::size_t i = 0; i + 1 < hd; i += 2) {
            const float theta = std::pow(10000.0f, -static_cast<float>(i) / static_cast<float>(hd));
            const float a = static_cast<float>(pos) * theta, c = std::cos(a), s = std::sin(a);
            auto rot = [&](std::vector<float>& z) {
                float x0 = z[b + i], x1 = z[b + i + 1];
                z[b + i] = x0 * c - x1 * s;
                z[b + i + 1] = x0 * s + x1 * c;
            };
            rot(q);
            rot(k);
        }
    }
}

class MiniLlama {
  public:
    explicit MiniLlama(const Config& c) : c_(c), lm_(c.vocab, c.hidden, 0xABCDEF) {
        if (c.hidden % 32 || c.ff % 32 || c.hidden % c.heads)
            throw std::runtime_error("invalid dimensions");
        layers_.reserve(c.layers);
        for (std::size_t i = 0; i < c.layers; ++i)
            layers_.emplace_back(c.hidden, c.ff, 0x10000 + i * 17);
        kcache_.resize(c.layers * c.max_ctx * c.hidden);
        vcache_.resize(c.layers * c.max_ctx * c.hidden);
        x_.resize(c.hidden);
        n_.resize(c.hidden);
        q_.resize(c.hidden);
        k_.resize(c.hidden);
        v_.resize(c.hidden);
        att_.resize(c.hidden);
        proj_.resize(c.hidden);
        gate_.resize(c.ff);
        up_.resize(c.ff);
        ff_.resize(c.ff);
        out_.resize(c.hidden);
        logits_.resize(c.vocab);
    }
    void reset() { pos_ = 0; }
    std::uint64_t params() const {
        std::uint64_t p = lm_.params();
        for (auto& l : layers_)
            p += l.params();
        return p;
    }
    std::uint64_t weight_bytes() const {
        std::uint64_t p = lm_.bytes();
        for (auto& l : layers_)
            p += l.bytes();
        return p;
    }
    std::size_t kv_bytes() const { return (kcache_.size() + vcache_.size()) * sizeof(float); }

    int step(int token, bool logits = true) {
        // Deterministic byte-like embedding; benchmark focuses decoder execution, not model
        // quality.
        for (std::size_t i = 0; i < c_.hidden; ++i)
            x_[i] = 0.03f * std::sin((token + 1) * (i + 1) * 0.0017f) +
                    0.01f * std::cos((token + 3) * (i + 7) * 0.0009f);
        const std::size_t hd = c_.hidden / c_.heads;
        for (std::size_t li = 0; li < c_.layers; ++li) {
            Layer& L = layers_[li];
            rmsnorm(n_, x_);
            L.q.mul(n_.data(), q_.data(), c_.threads);
            L.k.mul(n_.data(), k_.data(), c_.threads);
            L.v.mul(n_.data(), v_.data(), c_.threads);
            rope(q_, k_, pos_, c_.heads);
            float* kc = &kcache_[(li * c_.max_ctx + pos_) * c_.hidden];
            float* vc = &vcache_[(li * c_.max_ctx + pos_) * c_.hidden];
            std::memcpy(kc, k_.data(), c_.hidden * sizeof(float));
            std::memcpy(vc, v_.data(), c_.hidden * sizeof(float));
            std::fill(att_.begin(), att_.end(), 0.0f);
            std::vector<float> score(pos_ + 1);
            for (std::size_t head = 0; head < c_.heads; ++head) {
                const std::size_t hb = head * hd;
                float mx = -std::numeric_limits<float>::infinity();
                for (std::size_t t = 0; t <= pos_; ++t) {
                    const float* kt = &kcache_[(li * c_.max_ctx + t) * c_.hidden + hb];
                    float s = 0;
                    for (std::size_t d = 0; d < hd; ++d)
                        s += q_[hb + d] * kt[d];
                    s /= std::sqrt(static_cast<float>(hd));
                    score[t] = s;
                    mx = std::max(mx, s);
                }
                float den = 0;
                for (std::size_t t = 0; t <= pos_; ++t) {
                    score[t] = std::exp(score[t] - mx);
                    den += score[t];
                }
                const float inv = 1.0f / den;
                for (std::size_t t = 0; t <= pos_; ++t) {
                    const float a = score[t] * inv;
                    const float* vt = &vcache_[(li * c_.max_ctx + t) * c_.hidden + hb];
                    for (std::size_t d = 0; d < hd; ++d)
                        att_[hb + d] += a * vt[d];
                }
            }
            L.o.mul(att_.data(), proj_.data(), c_.threads);
            for (std::size_t i = 0; i < c_.hidden; ++i)
                x_[i] += proj_[i];
            rmsnorm(n_, x_);
            L.gate.mul(n_.data(), gate_.data(), c_.threads);
            L.up.mul(n_.data(), up_.data(), c_.threads);
            for (std::size_t i = 0; i < c_.ff; ++i)
                ff_[i] = silu(gate_[i]) * up_[i];
            L.down.mul(ff_.data(), out_.data(), c_.threads);
            for (std::size_t i = 0; i < c_.hidden; ++i)
                x_[i] += out_[i];
        }
        ++pos_;
        if (!logits)
            return token;
        rmsnorm(n_, x_);
        lm_.mul(n_.data(), logits_.data(), c_.threads);
        return static_cast<int>(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
    }
    std::size_t pos() const { return pos_; }

  private:
    Config c_;
    std::vector<Layer> layers_;
    Q8Matrix lm_;
    std::size_t pos_ = 0;
    std::vector<float> kcache_, vcache_, x_, n_, q_, k_, v_, att_, proj_, gate_, up_, ff_, out_,
        logits_;
};

struct Result {
    std::vector<double> pp_tps, tg_tps, ttft_ms, out_tps;
};

int main_impl(int argc, char** argv) {
    Config c;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value");
            return std::string(argv[++i]);
        };
        if (a == "--threads")
            c.threads = std::stoul(val());
        else if (a == "--reps")
            c.reps = std::stoul(val());
        else if (a == "--warmup")
            c.warmup = std::stoul(val());
        else if (a == "--hidden")
            c.hidden = std::stoull(val());
        else if (a == "--ff")
            c.ff = std::stoull(val());
        else if (a == "--layers")
            c.layers = std::stoull(val());
        else if (a == "--heads")
            c.heads = std::stoull(val());
        else if (a == "--csv")
            csv = val();
    }
    const auto build0 = Clock::now();
    MiniLlama m(c);
    const auto build1 = Clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(build1 - build0).count();
    auto pp = [&](std::size_t n) {
        m.reset();
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
            m.step(static_cast<int>((i * 37 + 11) % c.vocab), false);
        auto t1 = Clock::now();
        return n / std::chrono::duration<double>(t1 - t0).count();
    };
    auto tg = [&](std::size_t n) {
        m.reset();
        for (std::size_t i = 0; i < 128; ++i)
            m.step(static_cast<int>((i * 19 + 7) % c.vocab), false);
        int tok = 17;
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
            tok = m.step(tok, true);
        auto t1 = Clock::now();
        return n / std::chrono::duration<double>(t1 - t0).count();
    };
    auto client = [&]() {
        m.reset();
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < 128; ++i)
            m.step(static_cast<int>((i * 13 + 5) % c.vocab), false);
        int tok = m.step(23, true);
        auto t1 = Clock::now();
        for (std::size_t i = 1; i < 256; ++i)
            tok = m.step(tok, true);
        auto t2 = Clock::now();
        double tt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double tps = 255.0 / std::chrono::duration<double>(t2 - t1).count();
        return std::pair<double, double>(tt, tps);
    };
    for (unsigned i = 0; i < c.warmup; ++i) {
        (void)pp(32);
        (void)tg(8);
    }
    Result r;
    for (unsigned rep = 0; rep < c.reps; ++rep) {
        r.pp_tps.push_back(pp(512));
        r.tg_tps.push_back(tg(128));
        auto [tt, tps] = client();
        r.ttft_ms.push_back(tt);
        r.out_tps.push_back(tps);
    }
    const auto spp = calc(r.pp_tps), stg = calc(r.tg_tps), stt = calc(r.ttft_ms),
               sot = calc(r.out_tps);
    const double mib = m.weight_bytes() / 1048576.0, kv_mib = m.kv_bytes() / 1048576.0;
    std::cout << "| model | size | params | backend | threads | test | result "
                 "|\n|---|---:|---:|---|---:|---|---:|\n";
    std::cout << "| MemVanta Synthetic Llama Q8_0 | " << std::fixed << std::setprecision(2) << mib
              << " MiB | " << m.params() / 1e6 << " M | CPU/AVX2 | " << c.threads << " | pp512 | "
              << spp.mean << " ± " << spp.sd << " t/s |\n";
    std::cout << "| MemVanta Synthetic Llama Q8_0 | " << mib << " MiB | " << m.params() / 1e6
              << " M | CPU/AVX2 | " << c.threads << " | tg128 | " << stg.mean << " ± " << stg.sd
              << " t/s |\n";
    std::cout << "\nMLPerf-client-style (128 input / 256 output): TTFT " << stt.mean << " ± "
              << stt.sd << " ms; output TPS " << sot.mean << " ± " << sot.sd << " t/s\n";
    std::cout << "Load/init: " << load_ms << " ms; Q8 weight bytes: " << mib
              << " MiB; KV capacity: " << kv_mib << " MiB; peak RSS: " << peak_rss_kib() / 1024.0
              << " MiB\n";
    if (!csv.empty()) {
        std::ofstream f(csv);
        f << "rep,pp512_tps,tg128_tps,ttft_ms,output_tps\n";
        for (std::size_t i = 0; i < r.pp_tps.size(); ++i)
            f << i + 1 << "," << r.pp_tps[i] << "," << r.tg_tps[i] << "," << r.ttft_ms[i] << ","
              << r.out_tps[i] << "\n";
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
