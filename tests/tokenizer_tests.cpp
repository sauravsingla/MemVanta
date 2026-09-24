#include "check.hpp"
#include "memvanta/gguf.hpp"
#include "memvanta/llama_model.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void roundtrip(memvanta::GgufTokenizer& t, const std::string& s) {
    auto ids = t.encode(s, false);
    auto d = t.decode(ids);
    CHECK_MSG(d == s, ("encode/decode round-trip mismatch, model=" + t.model_type()).c_str());
}
static std::string ids_text(const std::vector<int>& ids) {
    std::string s = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i)
            s += ",";
        s += std::to_string(ids[i]);
    }
    return s + "]";
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tokenizer_tests <gpt2.gguf> <sp.gguf>\n";
        return 2;
    }
    memvanta::GgufFile gf(argv[1]), sf(argv[2]);
    memvanta::GgufTokenizer g(gf), sp(sf);
    CHECK_FATAL(g.model_type() == "gpt2");
    CHECK_FATAL(sp.model_type() == "llama");
    // Byte completeness, including NUL and malformed UTF-8.
    std::string all;
    for (int i = 0; i < 256; ++i)
        all.push_back(static_cast<char>(i));
    roundtrip(g, all);
    roundtrip(sp, all);
    roundtrip(g, std::string("\xF0\x28\x8C\x28\0\xFF", 6));
    roundtrip(sp, std::string("\xF0\x28\x8C\x28\0\xFF", 6));
    std::mt19937 rng(7);
    for (int k = 0; k < 200; ++k) {
        std::string s;
        int n = rng() % 129;
        for (int i = 0; i < n; ++i)
            s.push_back(static_cast<char>(rng() % 256));
        roundtrip(g, s);
        roundtrip(sp, s);
    }
    // GPT-2 fixture has explicit merges for these.
    auto hi = g.encode("hello world", false);
    std::vector<int> hi_ref = {262, 267};
    CHECK_MSG(hi == hi_ref,
              "GPT-2 merge table did not produce the expected ids for \"hello world\"");
    auto special = g.encode("<|im_start|>user\nhello<|im_end|>", false);
    CHECK_MSG(!special.empty() && special.front() == 1 && special.back() == 2,
              "GGUF special tokens were not preserved verbatim");
    // Exact Llama/GGUF SPM merge vectors. These follow the same highest-score adjacent-pair
    // merge semantics used by the pinned llama.cpp interoperability reference.
    struct C {
        std::string s;
        std::vector<int> ids;
    };
    std::vector<C> cs = {
        {"Hello world", {259, 344, 260, 263, 263, 261, 259, 272, 261, 269, 263, 268}},
        {"नमस्ते दुनिया", {259, 294, 281, 347, 303, 289, 259, 287, 288, 294, 311, 333, 274}},
        {"Café naïve résumé",
         {295, 262, 267, 276, 259, 265, 262, 313, 293, 354, 276, 264, 297, 276}},
        {"emoji 😀 🚀 ❤️", {259, 260, 279, 261, 283, 270, 259, 342, 259, 343, 259, 334, 341}},
        {"  spaces   and\ttabs",
         {259, 259, 308, 275, 298, 312, 259, 259, 353, 265, 268, 12, 306, 282, 264}},
        {"if (x < 10) { return x * x; }",
         {259, 270, 267, 259, 315, 277, 259, 321, 259, 319, 318, 316, 259,
          327, 259, 305, 307, 266, 299, 301, 259, 317, 301, 320, 259, 328}},
    };
    for (const auto& c : cs) {
        auto got = sp.encode(c.s, false);
        auto msg = "Llama/GGUF tokenizer parity mismatch: " + c.s + " got=" + ids_text(got) +
                   " expected=" + ids_text(c.ids);
        CHECK_MSG(got == c.ids, msg.c_str());
        roundtrip(sp, c.s);
    }
    CHECK(sp.encode("", false).empty());
    CHECK(g.encode("", false).empty());
    if (memvanta_test::failures()) {
        std::cerr << "FAILED\n";
        return 1;
    }
    std::cout << "tokenizer tests ok: 2 tokenizer families, 202 byte-fuzz cases, 6 exact "
                 "Llama/GGUF SPM vectors\n";
    return 0;
}
