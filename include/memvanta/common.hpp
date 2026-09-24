#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
namespace memvanta {
inline double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}
inline double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}
inline std::uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
std::uint64_t parse_size(const std::string& s);
} // namespace memvanta
