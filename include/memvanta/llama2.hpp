#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace memvanta {

struct Llama2Config {
    std::int32_t dim{};
    std::int32_t hidden_dim{};
    std::int32_t n_layers{};
    std::int32_t n_heads{};
    std::int32_t n_kv_heads{};
    std::int32_t vocab_size{};
    std::int32_t seq_len{};
};

struct Llama2BenchResult {
    double prompt_tps{};
    double generation_tps{};
    double ttft_ms{};
    double output_tps{};
    std::size_t peak_rss_kib{};
};

class Llama2Model {
  public:
    Llama2Model(const std::string& path, unsigned threads = 1);
    ~Llama2Model();
    Llama2Model(const Llama2Model&) = delete;
    Llama2Model& operator=(const Llama2Model&) = delete;

    const Llama2Config& config() const noexcept { return cfg_; }
    std::size_t file_size() const noexcept { return file_size_; }
    unsigned threads() const noexcept { return threads_; }
    void set_threads(unsigned n) noexcept { threads_ = n ? n : 1; }

    // Execute one decoder step at absolute position `pos` and return logits.
    const float* forward(int token, int pos);
    int greedy(int token, int pos);
    void clear_kv();

    // llama-bench-like end-to-end metrics using token IDs so tokenizer cost is excluded,
    // just as llama-bench focuses on model evaluation throughput.
    Llama2BenchResult
    benchmark(std::size_t prompt_tokens, std::size_t generated_tokens, unsigned warmup = 1);

  private:
    struct Weights;
    struct State;
    int fd_ = -1;
    void* map_ = nullptr;
    std::size_t file_size_ = 0;
    Llama2Config cfg_{};
    bool shared_classifier_ = true;
    unsigned threads_ = 1;
    Weights* w_ = nullptr;
    State* s_ = nullptr;

    void map_weights();
};

} // namespace memvanta
