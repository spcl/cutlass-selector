// MIT License

// Copyright (c) 2024-2026 HazyResearch

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <chrono>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <thread>

///////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
///////////////////////////////////////////////////////////////////////////////////////////////////

static inline void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Fill kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

enum class FillMode { CONSTANT, RANDOM };

template <typename T, FillMode mode>
__global__ void fill_kernel(T* data,
                            size_t count,
                            uint64_t seed,
                            float min_val = 0.,
                            float max_val = 0.) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val;
        if constexpr (mode == FillMode::CONSTANT) {
            val = min_val;
        } else if constexpr (mode == FillMode::RANDOM) {
            // Splitmix64 hash for uniform random bits
            uint64_t x = seed + idx;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            // Upper 24 bits to float in [0,1)
            float u = (float)(x >> 40) * (1.0f / 16777216.0f);
            // Scale to [min_val, max_val]
            val = u * (max_val - min_val) + min_val;
        }
        data[idx] = static_cast<T>(val);
    }
}

template <typename T, FillMode mode>
static inline void fill(T* data, size_t count, float value) {
    static_assert(mode == FillMode::CONSTANT, "Use fill<T, FillMode::CONSTANT> for constant fill");
    dim3 block(256);
    dim3 grid((count + 255) / 256);
    fill_kernel<T, mode><<<grid, block>>>(data, count, 0, value, 0.0f);
}

template <typename T, FillMode mode>
static inline void fill(T* data, size_t count, uint64_t seed, float min_val, float max_val) {
    static_assert(mode == FillMode::RANDOM, "Use fill<T, FillMode::RANDOM> for random fill");
    dim3 block(256);
    dim3 grid((count + 255) / 256);
    fill_kernel<T, mode><<<grid, block>>>(data, count, seed, min_val, max_val);
}
