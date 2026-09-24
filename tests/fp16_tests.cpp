#include "check.hpp"
#include "memvanta/gguf.hpp"
#include "memvanta/gguf_kernels.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
    using memvanta::fp16_to_fp32;
    using memvanta::fp32_to_fp16;

    CHECK_MSG(fp32_to_fp16(1.99951171875f) == 0x4000u,
              "FP16 mantissa rounding did not carry into the exponent at 2.0");
    CHECK_MSG(fp32_to_fp16(65504.0f) == 0x7bffu, "FP16 maximum finite value encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(65520.0f) == 0x7c00u,
              "FP16 overflow boundary did not round to infinity");
    CHECK_MSG(fp32_to_fp16(std::ldexp(1.0f, -14)) == 0x0400u,
              "FP16 minimum normal value encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(std::ldexp(1.0f, -24)) == 0x0001u,
              "FP16 minimum subnormal value encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(std::ldexp(1.0f, -25)) == 0x0000u,
              "FP16 half-way subnormal tie did not round to even zero");
    CHECK_MSG(fp32_to_fp16(0.0f) == 0x0000u, "positive zero encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(-0.0f) == 0x8000u, "negative zero encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(std::numeric_limits<float>::infinity()) == 0x7c00u,
              "positive infinity encoded incorrectly");
    CHECK_MSG(fp32_to_fp16(-std::numeric_limits<float>::infinity()) == 0xfc00u,
              "negative infinity encoded incorrectly");

    const auto nan_bits = fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    CHECK_MSG((nan_bits & 0x7c00u) == 0x7c00u && (nan_bits & 0x03ffu) != 0,
              "NaN did not remain NaN in FP16");

    for (std::uint32_t raw = 0; raw <= 0xffffu; ++raw) {
        const auto h = static_cast<std::uint16_t>(raw);
        const bool is_nan = (h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0;
        const auto roundtrip = fp32_to_fp16(fp16_to_fp32(h));
        if (is_nan) {
            CHECK_MSG((roundtrip & 0x7c00u) == 0x7c00u && (roundtrip & 0x03ffu) != 0,
                      "FP16 NaN round-trip lost NaN classification");
            CHECK_MSG((roundtrip & 0x8000u) == (h & 0x8000u), "FP16 NaN round-trip lost sign bit");
        } else {
            CHECK_MSG(roundtrip == h, "exhaustive FP16 round-trip mismatch");
        }
    }

    if (memvanta_test::failures()) {
        std::cerr << "FAILED\n";
        return 1;
    }
    std::cout << "fp16 tests ok: boundary rounding plus exhaustive 65536-value round-trip\n";
    return 0;
}
