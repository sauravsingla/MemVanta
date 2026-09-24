#pragma once
#include "memvanta/gguf.hpp"
#include "memvanta/worker_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace memvanta {

enum class KernelProfileKind { QProj, KProj, VProj, OProj, FfnGate, FfnUp, FfnDown, Output, Other };
struct KernelProfileEntry {
    int layer{-1};
    KernelProfileKind kind{KernelProfileKind::Other};
    std::string tensor;
    std::size_t calls{};
    std::size_t batch{};
    double ms{};
};
void enable_kernel_profiling(bool enabled);
void reset_kernel_profile();
std::vector<KernelProfileEntry> kernel_profile_snapshot();
const char* kernel_profile_kind_name(KernelProfileKind kind);

struct MatvecTarget {
    const GgufTensor* tensor{};
    float* output{};
};

void tensor_read_row_f32(const GgufFile&, const GgufTensor&, std::size_t, float*, std::size_t);
void tensor_matvec(const GgufFile&,
                   const GgufTensor&,
                   const float*,
                   float*,
                   unsigned threads = 1,
                   WorkerPool* pool = nullptr);
// Execute several projections over the same activation. When every target is Q4_0
// with the same input width, the activation is quantized to Q8 once and reused.
// Other type combinations safely fall back to tensor_matvec per target.
void tensor_matvec_group(const GgufFile&,
                         const MatvecTarget* targets,
                         std::size_t count,
                         const float* x,
                         unsigned threads = 1,
                         WorkerPool* pool = nullptr);
void tensor_matmul_batch(const GgufFile&,
                         const GgufTensor&,
                         const float* x,
                         float* y,
                         std::size_t batch,
                         unsigned threads = 1,
                         WorkerPool* pool = nullptr);
// Hybrid Q4/Q8 execution: batch==1 uses packed integer Q4 x Q8-activation SIMD;
// batched prefill keeps FP32 activations, with wider FFN blocking than Q/K/V/O.
void tensor_matmul_batch_v06(const GgufFile&,
                             const GgufTensor&,
                             const float* x,
                             float* y,
                             std::size_t batch,
                             unsigned threads = 1,
                             WorkerPool* pool = nullptr);
void tensor_vector_to_f32(const GgufFile&, const GgufTensor&, float*, std::size_t);
float dot_f32_simd(const float* a, const float* b, std::size_t n);
void axpy_f32_simd(float alpha, const float* x, float* y, std::size_t n);
std::uint16_t fp32_to_fp16(float x);
} // namespace memvanta
