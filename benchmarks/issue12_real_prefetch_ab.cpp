#include "memvanta/llama_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {
double rss_mib() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1024.0;
}
std::pair<long, long> faults() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return {r.ru_minflt, r.ru_majflt};
}
std::vector<int> fixed_tokens(std::size_t n, std::size_t vocab) {
    std::vector<int> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = static_cast<int>((i * 1543 + 17) % std::max<std::size_t>(vocab, 1));
    return x;
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2.0;
}

void drop_model_pages(const memvanta::LlamaModel& m) {
    const auto& mf = m.gguf().mapped_file();
    for (const auto& t : m.gguf().tensors())
        mf.advise_dontneed(t.offset, t.nbytes);
}

struct PrefetchResult {
    std::size_t layers{};
    std::uint64_t bytes{};
};
PrefetchResult
prefetch_hot_layers(const memvanta::LlamaModel& m, std::size_t depth, std::uint64_t byte_budget) {
    PrefetchResult out{};
    if (!depth || !byte_budget)
        return out;
    const auto& mf = m.gguf().mapped_file();
    const auto& ts = m.gguf().tensors();
    const std::size_t layers = std::min(depth, m.config().n_layer);
    for (std::size_t li = 0; li < layers; ++li) {
        const std::string p = "blk." + std::to_string(li) + ".";
        std::uint64_t layer_bytes = 0;
        for (const auto& t : ts)
            if (t.name.rfind(p, 0) == 0) {
                // Norm vectors are materialized by LlamaModel construction.  The
                // mapped hot set is the projection/FFN weight data only.
                if (t.name.find("_norm.weight") != std::string::npos)
                    continue;
                if (t.nbytes > byte_budget - layer_bytes) {
                    layer_bytes = byte_budget + 1;
                    break;
                }
                layer_bytes += t.nbytes;
            }
        if (layer_bytes > byte_budget || out.bytes > byte_budget - layer_bytes)
            break;
        for (const auto& t : ts)
            if (t.name.rfind(p, 0) == 0 && t.name.find("_norm.weight") == std::string::npos)
                mf.advise_willneed(t.offset, t.nbytes);
        out.bytes += layer_bytes;
        ++out.layers;
    }
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string model, mode = "none", csv;
        unsigned threads = 4, reps = 5;
        std::size_t ctx = 128, prompt_n = 32, gen_n = 8, depth = 2, min_depth = 1, max_depth = 4;
        std::uint64_t budget_mib = 128;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto val = [&]() {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--model")
                model = val();
            else if (a == "--mode")
                mode = val();
            else if (a == "--threads")
                threads = std::stoul(val());
            else if (a == "--reps")
                reps = std::stoul(val());
            else if (a == "--ctx")
                ctx = std::stoull(val());
            else if (a == "--prompt")
                prompt_n = std::stoull(val());
            else if (a == "--gen")
                gen_n = std::stoull(val());
            else if (a == "--depth")
                depth = std::stoull(val());
            else if (a == "--min-depth")
                min_depth = std::stoull(val());
            else if (a == "--max-depth")
                max_depth = std::stoull(val());
            else if (a == "--budget-mib")
                budget_mib = std::stoull(val());
            else if (a == "--csv")
                csv = val();
            else
                throw std::runtime_error("unknown arg: " + a);
        }
        if (model.empty())
            throw std::runtime_error("--model required");
        if (mode != "none" && mode != "fixed" && mode != "adaptive")
            throw std::runtime_error("--mode must be none|fixed|adaptive");
        if (!budget_mib || budget_mib > 16384)
            throw std::runtime_error("--budget-mib must be in 1..16384");
        const std::uint64_t byte_budget = budget_mib * 1024ull * 1024ull;
        memvanta::LlamaModel m(model, threads, ctx);
        prompt_n = std::min(prompt_n, m.config().n_ctx / 2);
        gen_n = std::min(gen_n, m.config().n_ctx / 4);
        auto prompt = fixed_tokens(prompt_n, m.config().vocab);
        auto seed =
            fixed_tokens(std::min<std::size_t>(16, m.config().n_ctx - gen_n), m.config().vocab);
        memvanta::Sampler greedy({0, 1, 1});
        std::vector<double> pp, tg, minflt;
        std::size_t current = std::clamp(depth, min_depth, max_depth), max_prefetched_layers = 0;
        std::uint64_t max_prefetch_bytes = 0;
        double prev_total = 0;
        long major_faults_total = 0;
        int final_token = 0;
        for (unsigned r = 0; r < reps; ++r) {
            drop_model_pages(m);
            PrefetchResult pref{};
            if (mode == "fixed")
                pref = prefetch_hot_layers(m, depth, byte_budget);
            else if (mode == "adaptive")
                pref = prefetch_hot_layers(m, current, byte_budget);
            max_prefetched_layers = std::max(max_prefetched_layers, pref.layers);
            max_prefetch_bytes = std::max(max_prefetch_bytes, pref.bytes);
            const auto f0 = faults();
            m.reset();
            auto a = Clock::now();
            m.prefill(prompt, 16, true);
            auto b = Clock::now();
            double pp_s = std::chrono::duration<double>(b - a).count();
            pp.push_back(prompt.size() / pp_s);
            m.reset();
            m.prefill(seed, 16, false);
            int cur = 17 % m.config().vocab;
            auto c = Clock::now();
            for (std::size_t i = 0; i < gen_n; ++i)
                cur = greedy.sample(m.forward(cur, true));
            auto d = Clock::now();
            final_token = cur;
            double tg_s = std::chrono::duration<double>(d - c).count();
            tg.push_back(gen_n / tg_s);
            const auto f1 = faults();
            minflt.push_back(double(std::max<long>(0, f1.first - f0.first)));
            major_faults_total += std::max<long>(0, f1.second - f0.second);
            const double total = pp_s + tg_s;
            if (mode == "adaptive" && r > 0) {
                if (total > prev_total * 1.03 && current > min_depth)
                    --current;
                else if (total < prev_total * 0.99 && current < max_depth)
                    ++current;
            }
            prev_total = total;
        }
        std::cout << "mode=" << mode << " threads=" << threads << " pp_tps_median=" << median(pp)
                  << " tg_tps_median=" << median(tg) << " peak_rss_mib=" << rss_mib()
                  << " final_depth=" << current << " budget_mib=" << budget_mib
                  << " max_prefetch_mib=" << (double(max_prefetch_bytes) / (1024.0 * 1024.0))
                  << " max_prefetched_layers=" << max_prefetched_layers
                  << " minor_faults_median=" << median(minflt)
                  << " major_faults_total=" << major_faults_total << " final_token=" << final_token
                  << "\n";
        if (!csv.empty()) {
            std::ofstream f(csv);
            f << "rep,pp_tps,tg_tps,minor_faults\n";
            for (std::size_t i = 0; i < pp.size(); ++i)
                f << i + 1 << "," << pp[i] << "," << tg[i] << "," << minflt[i] << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}