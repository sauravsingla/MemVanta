#pragma once
#include <cstddef>
namespace memvanta {
void matvec_f32(const float* matrix,
                const float* x,
                float* y,
                std::size_t rows,
                std::size_t cols,
                unsigned threads);
}
