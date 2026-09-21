#include "memvanta/quant.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(MEMVANTA_USE_OPENMP)
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace memvanta {
namespace {
template<class Fn>
void parallel_rows(std::size_t rows, unsigned threads, Fn fn) {
    if (!rows) return;
    threads = std::max(1u, threads);
    if (rows < threads) threads = static_cast<unsigned>(rows);
    if (threads == 1) { fn(0, rows); return; }
#if defined(MEMVANTA_USE_OPENMP)
    const std::size_t step = (rows + threads - 1) / threads;
    #pragma omp parallel num_threads(threads)
    {
        const unsigned t = static_cast<unsigned>(omp_get_thread_num());
        const std::size_t a = t*step;
        const std::size_t b = std::min(rows, a+step);
        if (a < b) fn(a,b);
    }
#else
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const std::size_t step = (rows + threads - 1) / threads;
    for (unsigned t=0; t<threads; ++t) {
        const std::size_t a = t*step;
        const std::size_t b = std::min(rows, a+step);
        if (a < b) pool.emplace_back(fn, a, b);
    }
    for (auto &th : pool) th.join();
#endif
}
}

std::vector<BlockQ8_0> quantize_q8_0(const float* src, std::size_t n) {
    if (n % QK) throw std::runtime_error("q8_0 length must be multiple of 32");
    std::vector<BlockQ8_0> out(n/QK);
    for (std::size_t b=0; b<out.size(); ++b) {
        const float* p = src + b*QK;
        float amax = 0.0f;
        for (std::size_t i=0;i<QK;++i) amax = std::max(amax, std::fabs(p[i]));
        const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
        out[b].d = d;
        if (d == 0.0f) { std::fill(std::begin(out[b].qs), std::end(out[b].qs), 0); continue; }
        const float id = 1.0f/d;
        for (std::size_t i=0;i<QK;++i) {
            int q = static_cast<int>(std::nearbyint(p[i]*id));
            q = std::clamp(q, -127, 127);
            out[b].qs[i] = static_cast<std::int8_t>(q);
        }
    }
    return out;
}

std::vector<BlockQ4_0> quantize_q4_0(const float* src, std::size_t n) {
    if (n % QK) throw std::runtime_error("q4_0 length must be multiple of 32");
    std::vector<BlockQ4_0> out(n/QK);
    for (std::size_t b=0; b<out.size(); ++b) {
        const float* p = src + b*QK;
        float amax = 0.0f;
        for (std::size_t i=0;i<QK;++i) amax = std::max(amax, std::fabs(p[i]));
        const float d = amax == 0.0f ? 0.0f : amax / 7.0f;
        out[b].d = d;
        for (std::size_t i=0;i<QK/2;++i) out[b].qs[i] = 0;
        if (d == 0.0f) continue;
        const float id = 1.0f/d;
        for (std::size_t i=0;i<QK;++i) {
            int q = static_cast<int>(std::nearbyint(p[i]*id));
            q = std::clamp(q, -8, 7);
            const std::uint8_t nib = static_cast<std::uint8_t>(q + 8);
            if ((i & 1u)==0) out[b].qs[i/2] = nib;
            else out[b].qs[i/2] |= static_cast<std::uint8_t>(nib << 4);
        }
    }
    return out;
}

float dot_q8_0(const BlockQ8_0* a, const float* x, std::size_t n) {
    const std::size_t nb = n/QK;
    float sum = 0.0f;
    for (std::size_t b=0;b<nb;++b) {
#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (int k=0;k<32;k+=8) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a[b].qs+k));
            __m256i qi = _mm256_cvtepi8_epi32(q8);
            __m256 qf = _mm256_cvtepi32_ps(qi);
            __m256 xv = _mm256_loadu_ps(x+b*QK+k);
            acc = _mm256_fmadd_ps(qf, xv, acc);
        }
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc);
        float s=0; for(float v:tmp) s+=v;
        sum += s*a[b].d;
#else
        float s=0; for(std::size_t k=0;k<QK;++k) s += static_cast<float>(a[b].qs[k])*x[b*QK+k];
        sum += s*a[b].d;
#endif
    }
    return sum;
}

float dot_q4_0(const BlockQ4_0* a, const float* x, std::size_t n) {
    const std::size_t nb = n/QK;
    float sum=0.0f;
    for (std::size_t b=0;b<nb;++b) {
#if defined(__AVX2__)
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a[b].qs));
        const __m128i mask = _mm_set1_epi8(0x0f);
        const __m128i lo = _mm_and_si128(packed, mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        const __m128i u0 = _mm_unpacklo_epi8(lo, hi);
        const __m128i u1 = _mm_unpackhi_epi8(lo, hi);
        const __m128i bias = _mm_set1_epi8(8);
        const __m128i q0 = _mm_sub_epi8(u0, bias);
        const __m128i q1 = _mm_sub_epi8(u1, bias);
        __m256 acc = _mm256_setzero_ps();
        const __m256i q00 = _mm256_cvtepi8_epi32(q0);
        const __m256i q08 = _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8));
        const __m256i q10 = _mm256_cvtepi8_epi32(q1);
        const __m256i q18 = _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q00), _mm256_loadu_ps(x+b*QK), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q08), _mm256_loadu_ps(x+b*QK+8), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q10), _mm256_loadu_ps(x+b*QK+16), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q18), _mm256_loadu_ps(x+b*QK+24), acc);
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc);
        float s=0.0f; for(float v:tmp) s+=v;
        sum += s*a[b].d;
#else
        float s=0.0f;
        for (std::size_t k=0;k<QK;++k) {
            const std::uint8_t packed = a[b].qs[k/2];
            const int q = ((k&1u) ? (packed>>4) : (packed&0x0F)) - 8;
            s += static_cast<float>(q)*x[b*QK+k];
        }
        sum += s*a[b].d;
#endif
    }
    return sum;
}

void matvec_q8_0(const BlockQ8_0* A, const float* x, float* y,
                 std::size_t rows, std::size_t cols, unsigned threads) {
    if (cols % QK) throw std::runtime_error("q8_0 cols must be multiple of 32");
    const std::size_t bpr = cols/QK;
    parallel_rows(rows, threads, [&](std::size_t r0, std::size_t r1){
        for (std::size_t r=r0;r<r1;++r) y[r] = dot_q8_0(A+r*bpr, x, cols);
    });
}

void matvec_q4_0(const BlockQ4_0* A, const float* x, float* y,
                 std::size_t rows, std::size_t cols, unsigned threads) {
    if (cols % QK) throw std::runtime_error("q4_0 cols must be multiple of 32");
    const std::size_t bpr = cols/QK;
    parallel_rows(rows, threads, [&](std::size_t r0, std::size_t r1){
        for (std::size_t r=r0;r<r1;++r) y[r] = dot_q4_0(A+r*bpr, x, cols);
    });
}
}
