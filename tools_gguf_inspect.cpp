#include "memvanta/gguf.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static void usage(std::ostream& out) {
    out << "usage: memvanta_gguf_inspect <model.gguf> [--expect-arch <architecture>]\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            const std::string arg = argv[1];
            if (arg == "--help" || arg == "-h") {
                usage(std::cout);
                return 0;
            }
        }

        if (argc < 2 || argc > 4) {
            usage(std::cerr);
            return 2;
        }

        std::string expected;
        if (argc > 2) {
            if (argc != 4 || std::string(argv[2]) != "--expect-arch") {
                usage(std::cerr);
                return 2;
            }
            expected = argv[3];
        }

        memvanta::GgufFile file(argv[1]);
        const auto architecture = file.get_string("general.architecture").value_or("");
        if (architecture.empty())
            throw std::runtime_error("GGUF is missing general.architecture");
        if (!expected.empty() && architecture != expected) {
            throw std::runtime_error("unexpected GGUF architecture: expected " + expected +
                                     ", got " + architecture);
        }

        std::cout << "version=" << file.version() << '\n';
        std::cout << "architecture=" << architecture << '\n';
        std::cout << "metadata_items=" << file.metadata().size() << '\n';
        std::cout << "tensors=" << file.tensors().size() << '\n';
        std::cout << "data_offset=" << file.data_offset() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
