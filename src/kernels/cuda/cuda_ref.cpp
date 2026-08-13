#include "rapidllm/kernels/nv_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rapidllm::nv {

const char* nv_kernel_name() { return "gemv_dp4a (host-ref; CUDA twin in gemv_dp4a.cu)"; }

void pack_weight_int8(const float* W, signed char* out, float* row_scale, int m, int n) {
    for (int i = 0; i < m; ++i) {
        float amax = 0.f;
        const float* row = W + static_cast<size_t>(i) * n;
        for (int j = 0; j < n; ++j) amax = std::max(amax, std::fabs(row[j]));
        const float s = amax > 0.f ? amax / 127.f : 1.f;
        row_scale[i] = s;
        for (int j = 0; j < n; ++j) {
            int q = static_cast<int>(std::lround(row[j] / s));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out[static_cast<size_t>(i) * n + j] = static_cast<signed char>(q);
        }
    }
}

// Host twin of __dp4a: four int8 MACs per group (same reduction as sm_61 dp4a).
static inline int dp4a_host(int a, int b, int c) {
    const auto s8 = [](int packed, int sh) -> int {
        return static_cast<int>(static_cast<int8_t>((packed >> sh) & 0xFF));
    };
    return c + s8(a, 0) * s8(b, 0) + s8(a, 8) * s8(b, 8) + s8(a, 16) * s8(b, 16) +
           s8(a, 24) * s8(b, 24);
}

void gemv_dp4a_ref(const signed char* W, const float* row_scale, const float* x, float* y, int m,
                   int n) {
    float amax = 0.f;
    for (int j = 0; j < n; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float xs = amax > 0.f ? amax / 127.f : 1.f;
    std::vector<int8_t> xq(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        int q = static_cast<int>(std::lround(x[j] / xs));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        xq[j] = static_cast<int8_t>(q);
    }
    const int n4 = n & ~3;
    for (int i = 0; i < m; ++i) {
        int acc = 0;
        const signed char* row = W + static_cast<size_t>(i) * n;
        int j = 0;
        for (; j < n4; j += 4) {
            int wa = 0, xa = 0;
            for (int k = 0; k < 4; ++k) {
                wa |= (static_cast<unsigned char>(row[j + k]) << (8 * k));
                xa |= (static_cast<unsigned char>(xq[j + k]) << (8 * k));
            }
            acc = dp4a_host(wa, xa, acc);
        }
        for (; j < n; ++j) acc += static_cast<int>(row[j]) * static_cast<int>(xq[j]);
        y[i] = static_cast<float>(acc) * row_scale[i] * xs;
    }
}

#if !defined(RAPIDLLM_WITH_CUDA)
bool gemv_dp4a_cuda(const signed char*, const float*, const float*, float*, int, int) { return false; }
#endif

} // namespace rapidllm::nv
