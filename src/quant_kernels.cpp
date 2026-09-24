#include "memvanta/quant.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(MEMVANTA_USE_OPENMP)
#include <omp.h>
#endif

namespace memvanta {
namespace {
template <class Fn> void parallel_rows(std::size_t rows, unsigned threads, Fn fn) {
    threads = std::max(1u, threads);
    threads = std::min<unsigned>(threads, static_cast<unsigned>(std::max<std::size_t>(rows, 1)));
    if (threads == 1) {
        fn(0, rows);
        return;
    }
#if defined(MEMVANTA_USE_OPENMP)
    const std::size_t step = (rows + threads - 1) / threads;
#pragma omp parallel num_threads(threads)
    {
        const unsigned t = static_cast<unsigned>(omp_get_thread_num());
        const std::size_t a = t * step;
        const std::size_t b = std::min(rows, a + step);
        if (a < b)
            fn(a, b);
    }
#else
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const std::size_t step = (rows + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        const std::size_t a = t * step;
        const std::size_t b = std::min(rows, a + step);
        if (a < b)
            pool.emplace_back(fn, a, b);
    }
    for (auto& th : pool)
        th.join();
#endif
}
} // namespace

unsigned effective_kernel_threads(std::size_t rows,
                                  std::size_t cols,
                                  unsigned requested,
                                  std::size_t min_work_per_thread) {
    requested = std::max(1u, requested);
    if (requested == 1 || rows == 0 || cols == 0)
        return 1;
    const std::size_t work =
        rows > (static_cast<std::size_t>(-1) / cols) ? static_cast<std::size_t>(-1) : rows * cols;
    const std::size_t useful = min_work_per_thread ? work / min_work_per_thread : requested;
    return std::max(1u,
                    std::min<unsigned>(requested,
                                       static_cast<unsigned>(std::min<std::size_t>(
                                           std::max<std::size_t>(1, useful), rows))));
}

float dot_q4_q8(const BlockQ4_0* a, const BlockQ8_0* x, std::size_t n) {
    if (n % QK)
        throw std::runtime_error("q4_q8 length must be multiple of 32");
    const std::size_t nb = n / QK;
    float sum = 0.0f;
    for (std::size_t b = 0; b < nb; ++b) {
        int isum = 0;
        for (std::size_t k = 0; k < QK; ++k) {
            const std::uint8_t packed = a[b].qs[k / 2];
            const int q4 = ((k & 1u) ? (packed >> 4) : (packed & 0x0f)) - 8;
            isum += q4 * static_cast<int>(x[b].qs[k]);
        }
        sum += static_cast<float>(isum) * a[b].d * x[b].d;
    }
    return sum;
}

void matvec_q4_q8(const BlockQ4_0* A,
                  const BlockQ8_0* x,
                  float* y,
                  std::size_t rows,
                  std::size_t cols,
                  unsigned threads) {
    if (cols % QK)
        throw std::runtime_error("q4_q8 cols must be multiple of 32");
    const std::size_t bpr = cols / QK;
    const unsigned use_threads = effective_kernel_threads(rows, cols, threads);
    parallel_rows(rows, use_threads, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r)
            y[r] = dot_q4_q8(A + r * bpr, x, cols);
    });
}

void matmul_q4_0_batch(const BlockQ4_0* A,
                       const float* x,
                       float* y,
                       std::size_t rows,
                       std::size_t cols,
                       std::size_t batch,
                       unsigned threads) {
    if (cols % QK)
        throw std::runtime_error("q4_0 cols must be multiple of 32");
    const std::size_t bpr = cols / QK;
    const unsigned use_threads =
        effective_kernel_threads(rows, cols * std::max<std::size_t>(batch, 1), threads);
    parallel_rows(rows, use_threads, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const auto* row = A + r * bpr;
            for (std::size_t b = 0; b < batch; ++b)
                y[b * rows + r] = dot_q4_0(row, x + b * cols, cols);
        }
    });
}
} // namespace memvanta
