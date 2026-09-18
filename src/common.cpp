#include "memvanta/common.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace memvanta {

// Declared in common.hpp and used by both the CLI and the test suite, so it belongs
// in memvanta_core rather than in src/main.cpp: a definition that lives only in an
// executable cannot be linked by memvanta_tests.
std::uint64_t parse_size(const std::string& s) {
    if (s.empty()) throw std::runtime_error("empty size");
    std::size_t idx = 0;
    double v = std::stod(s, &idx);
    std::string suf = s.substr(idx);
    std::transform(suf.begin(), suf.end(), suf.begin(), [](unsigned char c){ return std::tolower(c); });
    double m = 1;
    if (suf == "k" || suf == "kb" || suf == "kib") m = 1024.;
    else if (suf == "m" || suf == "mb" || suf == "mib") m = 1024.*1024.;
    else if (suf == "g" || suf == "gb" || suf == "gib") m = 1024.*1024.*1024.;
    else if (!suf.empty() && suf != "b") throw std::runtime_error("bad size suffix: " + suf);
    return static_cast<std::uint64_t>(v*m);
}

} // namespace memvanta
