#include "memvanta/llama2.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

struct S {
    double mean{}, sd{};
};
static S stats(const std::vector<double>& v) {
    S s{};
    if (v.empty())
        return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    if (v.size() > 1) {
        double q = 0;
        for (double x : v)
            q += (x - s.mean) * (x - s.mean);
        s.sd = std::sqrt(q / (v.size() - 1));
    }
    return s;
}

int main(int argc, char** argv) {
    try {
        std::string model;
        unsigned threads = std::max(1u, std::thread::hardware_concurrency()), reps = 5, warmup = 1;
        std::size_t pp = 512, tg = 128;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto val = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--model")
                model = val();
            else if (a == "--threads")
                threads = std::stoul(val());
            else if (a == "--pp")
                pp = std::stoull(val());
            else if (a == "--tg")
                tg = std::stoull(val());
            else if (a == "--reps")
                reps = std::stoul(val());
            else if (a == "--warmup")
                warmup = std::stoul(val());
            else if (a == "--help") {
                std::cout << "usage: memvanta_llama2c_bench --model checkpoint.bin [--threads N "
                             "--pp 512 --tg 128 --reps 5 --warmup 1]\n";
                return 0;
            } else
                throw std::runtime_error("unknown arg: " + a);
        }
        if (model.empty())
            throw std::runtime_error("--model required");
        memvanta::Llama2Model m(model, threads);
        pp = std::min<std::size_t>(pp, m.config().seq_len);
        tg = std::min<std::size_t>(tg, m.config().seq_len);
        for (unsigned i = 0; i < warmup; ++i)
            (void)m.benchmark(std::min<std::size_t>(8, pp), std::min<std::size_t>(8, tg), 0);
        std::vector<double> pv, gv, tv, ov;
        std::size_t rss = 0;
        for (unsigned i = 0; i < reps; ++i) {
            auto r = m.benchmark(pp, tg, 0);
            pv.push_back(r.prompt_tps);
            gv.push_back(r.generation_tps);
            tv.push_back(r.ttft_ms);
            ov.push_back(r.output_tps);
            rss = std::max(rss, r.peak_rss_kib);
        }
        auto ps = stats(pv), gs = stats(gv), ts = stats(tv), os = stats(ov);
        std::cout << "MemVanta v0.4 llama2.c-checkpoint full-graph CPU benchmark\n";
        std::cout << "model_bytes=" << m.file_size() << " dim=" << m.config().dim
                  << " layers=" << m.config().n_layers << " heads=" << m.config().n_heads
                  << " kv_heads=" << m.config().n_kv_heads << " vocab=" << m.config().vocab_size
                  << " ctx=" << m.config().seq_len << " threads=" << threads << " reps=" << reps
                  << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "pp" << pp << " = " << ps.mean << " ± " << ps.sd << " tok/s\n";
        std::cout << "tg" << tg << " = " << gs.mean << " ± " << gs.sd << " tok/s\n";
        std::cout << "TTFT(<=128 prompt) = " << ts.mean << " ± " << ts.sd << " ms\n";
        std::cout << "output TPS = " << os.mean << " ± " << os.sd << " tok/s\n";
        std::cout << "peak RSS = " << (rss / 1024.0) << " MiB\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
