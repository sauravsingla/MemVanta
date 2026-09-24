#include "check.hpp"
#include "memvanta/gguf.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void write_header(std::ofstream& f, std::uint64_t tensors, std::uint64_t metadata) {
    f.write("GGUF", 4);
    const std::uint32_t version = 3;
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    f.write(reinterpret_cast<const char*>(&tensors), sizeof(tensors));
    f.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
}
void write_string(std::ofstream& f, const std::string& s) {
    const std::uint64_t n = s.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}
} // namespace

int main() {
    {
        const char* path = "memvanta_gguf_string_limit.gguf";
        std::ofstream f(path, std::ios::binary);
        write_header(f, 0, 1);
        write_string(f, "abcde");
        const std::uint32_t type = static_cast<std::uint32_t>(memvanta::GgufValueType::UInt8);
        const std::uint8_t value = 1;
        f.write(reinterpret_cast<const char*>(&type), sizeof(type));
        f.write(reinterpret_cast<const char*>(&value), sizeof(value));
        f.close();
        memvanta::GgufParseLimits limits;
        limits.max_string_bytes = 4;
        bool threw = false;
        try {
            memvanta::GgufFile g(path, limits);
            (void)g;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_MSG(threw, "GGUF string limit was not enforced");
        std::remove(path);
    }
    {
        const char* path = "memvanta_gguf_array_limit.gguf";
        std::ofstream f(path, std::ios::binary);
        write_header(f, 0, 1);
        write_string(f, "a");
        const std::uint32_t type = static_cast<std::uint32_t>(memvanta::GgufValueType::Array);
        const std::uint32_t elem = static_cast<std::uint32_t>(memvanta::GgufValueType::UInt8);
        const std::uint64_t count = 2;
        const std::uint8_t values[2] = {1, 2};
        f.write(reinterpret_cast<const char*>(&type), sizeof(type));
        f.write(reinterpret_cast<const char*>(&elem), sizeof(elem));
        f.write(reinterpret_cast<const char*>(&count), sizeof(count));
        f.write(reinterpret_cast<const char*>(values), sizeof(values));
        f.close();
        memvanta::GgufParseLimits limits;
        limits.max_array_elements = 1;
        bool threw = false;
        try {
            memvanta::GgufFile g(path, limits);
            (void)g;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_MSG(threw, "GGUF array element limit was not enforced");
        std::remove(path);
    }
    {
        const char* path = "memvanta_gguf_heap_limit.gguf";
        std::ofstream f(path, std::ios::binary);
        write_header(f, 0, 1);
        write_string(f, "a");
        const std::uint32_t type = static_cast<std::uint32_t>(memvanta::GgufValueType::UInt8);
        const std::uint8_t value = 1;
        f.write(reinterpret_cast<const char*>(&type), sizeof(type));
        f.write(reinterpret_cast<const char*>(&value), sizeof(value));
        f.close();
        memvanta::GgufParseLimits limits;
        limits.max_parse_heap_bytes = 64;
        bool threw = false;
        try {
            memvanta::GgufFile g(path, limits);
            (void)g;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK_MSG(threw, "GGUF parse heap budget was not enforced");
        std::remove(path);
    }
    if (memvanta_test::failures()) {
        std::cerr << "FAILED\n";
        return 1;
    }
    std::cout << "gguf parser limit tests ok\n";
    return 0;
}
