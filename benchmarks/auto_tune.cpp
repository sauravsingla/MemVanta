#include "memvanta/llama_model.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
static std::vector<int> tokens(std::size_t n, std::size_t vocab) {
    std::vector<int> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = int((i * 1543 + 17) % vocab);
    return x;
}
int main(int argc, char** argv) {
    try {
        std::string model;
        unsigned max_threads = std::max(1u, std::thread::hardware_concurrency());
        std::size_t tune_prompt = 128, ctx = 256;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto v = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value");
                return std::string(argv[++i]);
            };
            if (a == "--model")
                model = v();
            else if (a == "--max-threads")
                max_threads = std::stoul(v());
            else if (a == "--prompt")
                tune_prompt = std::stoull(v());
            else if (a == "--ctx")
                ctx = std::stoull(v());
            else
                throw std::runtime_error("unknown arg: " + a);
        }
        if (model.empty())
            throw std::runtime_error("--model required");
        struct R {
            unsigned th;
            std::size_t batch;
            double tps;
        };
        std::vector<R> rs;
        std::vector<std::size_t> batches = {8, 16, 32, 64};
        std::cout << "threads,batch,pp_tps\n";
        for (unsigned th = 1; th <= max_threads; ++th) {
            memvanta::LlamaModel m(model, th, ctx, 128, memvanta::KVCacheType::F16);
            auto p = tokens(std::min(tune_prompt, m.config().n_ctx), m.config().vocab);
            for (auto b : batches) {
                if (b > p.size() * 2)
                    continue;
                m.reset();
                m.prefill(
                    std::vector<int>(p.begin(), p.begin() + std::min<std::size_t>(8, p.size())),
                    std::min<std::size_t>(8, b),
                    false);
                m.reset();
                auto t0 = Clock::now();
                m.prefill(p, b, false);
                auto t1 = Clock::now();
                double tps = p.size() / std::chrono::duration<double>(t1 - t0).count();
                rs.push_back({th, b, tps});
                std::cout << th << "," << b << "," << tps << "\n";
            }
        }
        auto best = *std::max_element(
            rs.begin(), rs.end(), [](const R& a, const R& b) { return a.tps < b.tps; });
        std::cerr << "MemVanta v0.6 auto-tune: threads=" << best.th << " batch=" << best.batch
                  << " pp=" << best.tps << " tok/s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
