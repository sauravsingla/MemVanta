#include "memvanta/llama_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>

static double nll_of(const std::vector<float>& logits, int target) {
    float mx = *std::max_element(logits.begin(), logits.end());
    double z = 0;
    for (float x : logits)
        z += std::exp(double(x - mx));
    return std::log(z) + mx - logits.at((size_t)target);
}
int main(int argc, char** argv) {
    try {
        std::string model, text_file,
            prompt = "Once upon a time, there was a small robot who wanted to learn.";
        unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        size_t max_tokens = 512, ctx = 1024;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto v = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value");
                return std::string(argv[++i]);
            };
            if (a == "--model")
                model = v();
            else if (a == "--text")
                text_file = v();
            else if (a == "--prompt")
                prompt = v();
            else if (a == "--threads")
                threads = std::stoul(v());
            else if (a == "--max-tokens")
                max_tokens = std::stoull(v());
            else if (a == "--ctx")
                ctx = std::stoull(v());
            else
                throw std::runtime_error("unknown arg: " + a);
        }
        if (model.empty())
            throw std::runtime_error("--model required");
        if (!text_file.empty()) {
            std::ifstream f(text_file);
            if (!f)
                throw std::runtime_error("cannot open text file");
            prompt.assign(std::istreambuf_iterator<char>(f), {});
        }
        memvanta::LlamaModel m(model, threads, ctx, 128, memvanta::KVCacheType::F16);
        auto ids = m.tokenizer().encode(prompt, true);
        if (ids.size() < 2)
            throw std::runtime_error("text encoded to <2 tokens");
        if (ids.size() > max_tokens)
            ids.resize(max_tokens);
        if (ids.size() > m.config().n_ctx)
            ids.resize(m.config().n_ctx);
        m.reset();
        double nll = 0;
        size_t scored = 0;
        std::vector<int> generated;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            const auto& logits = m.forward(ids[i], true);
            nll += nll_of(logits, ids[i + 1]);
            ++scored;
        }
        double avg = nll / scored, ppl = std::exp(std::min(avg, 50.0));
        m.reset();
        memvanta::Sampler greedy({0, 1, 42});
        auto seed = m.tokenizer().encode("Once upon a time", true);
        auto out = m.generate(seed, 16, greedy);
        std::cout << std::setprecision(10) << "tokens=" << ids.size() << " scored=" << scored
                  << " nll=" << nll << " avg_nll=" << avg << " ppl=" << ppl << "\n";
        std::cout << "greedy_ids=";
        for (size_t i = 0; i < out.size(); ++i) {
            if (i)
                std::cout << ",";
            std::cout << out[i];
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
