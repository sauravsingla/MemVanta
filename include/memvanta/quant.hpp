#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace memvanta {
constexpr std::size_t QK = 32;
#pragma pack(push, 1)
struct BlockQ8_0 {
    float d;
    std::int8_t qs[QK];
};
struct BlockQ4_0 {
    float d;
    std::uint8_t qs[QK / 2];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 36);
static_assert(sizeof(BlockQ4_0) == 20);

std::vector<BlockQ8_0> quantize_q8_0(const float* src, std::size_t n);
std::vector<BlockQ4_0> quantize_q4_0(const float* src, std::size_t n);
float dot_q8_0(const BlockQ8_0* a, const float* x, std::size_t n);
float dot_q4_0(const BlockQ4_0* a, const float* x, std::size_t n);
float dot_q4_q8(const BlockQ4_0* a, const BlockQ8_0* x, std::size_t n);
unsigned effective_kernel_threads(std::size_t rows,
                                  std::size_t cols,
                                  unsigned requested,
                                  std::size_t min_work_per_thread = 32768);
void matvec_q8_0(const BlockQ8_0* A,
                 const float* x,
                 float* y,
                 std::size_t rows,
                 std::size_t cols,
                 unsigned threads = 1);
void matvec_q4_0(const BlockQ4_0* A,
                 const float* x,
                 float* y,
                 std::size_t rows,
                 std::size_t cols,
                 unsigned threads = 1);
void matvec_q4_q8(const BlockQ4_0* A,
                  const BlockQ8_0* x,
                  float* y,
                  std::size_t rows,
                  std::size_t cols,
                  unsigned threads = 1);
void matmul_q4_0_batch(const BlockQ4_0* A,
                       const float* x,
                       float* y,
                       std::size_t rows,
                       std::size_t cols,
                       std::size_t batch,
                       unsigned threads = 1);
} // namespace memvanta
