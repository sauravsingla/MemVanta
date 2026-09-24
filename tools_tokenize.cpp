#include "memvanta/gguf.hpp"
#include "memvanta/llama_model.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

static std::string hex(const std::string& s) {
    std::ostringstream o;
    o << std::hex << std::setfill('0');
    for (unsigned char c : s)
        o << std::setw(2) << static_cast<unsigned>(c);
    return o.str();
}
static std::string unhex(const std::string& h) {
    if (h.size() % 2)
        throw std::runtime_error("hex input must have even length");
    std::string s;
    s.reserve(h.size() / 2);
    for (std::size_t i = 0; i < h.size(); i += 2) {
        unsigned v = 0;
        std::istringstream is(h.substr(i, 2));
        is >> std::hex >> v;
        if (is.fail())
            throw std::runtime_error("invalid hex input");
        s.push_back(static_cast<char>(v));
    }
    return s;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: memvanta_tokenize <tokenizer.gguf> <text>|--hex-input <hex> [--bos]\n";
        return 2;
    }
    try {
        std::string text;
        int opt = 3;
        if (std::string(argv[2]) == "--hex-input") {
            if (argc < 4)
                throw std::runtime_error("--hex-input requires data");
            text = unhex(argv[3]);
            opt = 4;
        } else
            text = argv[2];
        bool bos = argc > opt && std::string(argv[opt]) == "--bos";
        memvanta::GgufFile f(argv[1]);
        memvanta::GgufTokenizer t(f);
        auto ids = t.encode(text, bos);
        auto dec = t.decode(ids);
        std::cout << "model=" << t.model_type() << " pre=" << t.pre_type() << "\nids=";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << "\ndecoded_hex=" << hex(dec) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
