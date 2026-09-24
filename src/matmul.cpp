#include "memvanta/matmul.hpp"

#include <algorithm>
#include <thread>
#include <vector>
namespace memvanta {
void matvec_f32(const float* A,
                const float* x,
                float* y,
                std::size_t rows,
                std::size_t cols,
                unsigned threads) {
    threads = std::max(1u, threads);
    threads = std::min<unsigned>(threads, rows);
    auto worker = [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const float* row = A + r * cols;
            float s = 0;
#pragma GCC ivdep
            for (std::size_t c = 0; c < cols; ++c)
                s += row[c] * x[c];
            y[r] = s;
        }
    };
    if (threads == 1) {
        worker(0, rows);
        return;
    }
    std::vector<std::thread> ts;
    std::size_t step = (rows + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        auto a = t * step, b = std::min(rows, a + step);
        if (a < b)
            ts.emplace_back(worker, a, b);
    }
    for (auto& th : ts)
        th.join();
}
} // namespace memvanta
