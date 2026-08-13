#include "rapidllm/kernels/nv_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int M = 4, N = 8;
    const float W[M * N] = {
        0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f, 0.2f, 0.4f, 0.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
        -0.5f, 0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 0.25f, -0.25f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    };
    const float x[N] = {1.f, 2.f, -1.f, 0.5f, 0.f, 1.f, -2.f, 0.25f};
    std::vector<signed char> Wi(M * N);
    std::vector<float> scale(M), y(M), yref(M);
    rapidllm::nv::pack_weight_int8(W, Wi.data(), scale.data(), M, N);
    rapidllm::nv::gemv_dp4a_ref(Wi.data(), scale.data(), x, y.data(), M, N);

    // Independent check: dequant W and dot in float (same pack contract).
    for (int i = 0; i < M; ++i) {
        float acc = 0.f;
        for (int j = 0; j < N; ++j) acc += static_cast<float>(Wi[i * N + j]) * scale[i] * x[j];
        yref[i] = acc;
    }
    // host-ref quantizes x too — compare against that contract, not full-precision W.
    float amax = 0.f;
    for (int j = 0; j < N; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float xs = amax / 127.f;
    for (int i = 0; i < M; ++i) {
        int acc = 0;
        for (int j = 0; j < N; ++j) {
            int xq = static_cast<int>(std::lround(x[j] / xs));
            if (xq > 127) xq = 127;
            if (xq < -127) xq = -127;
            acc += static_cast<int>(Wi[i * N + j]) * xq;
        }
        yref[i] = static_cast<float>(acc) * scale[i] * xs;
    }

    std::printf("nv_kernel=%s\n", rapidllm::nv::nv_kernel_name());
    for (int i = 0; i < M; ++i) {
        std::printf("y[%d]=%.6f ref=%.6f\n", i, y[i], yref[i]);
        if (std::fabs(y[i] - yref[i]) > 1e-4f) {
            std::fprintf(stderr, "host-ref mismatch at %d\n", i);
            return 1;
        }
    }
    const bool cuda = rapidllm::nv::gemv_dp4a_cuda(Wi.data(), scale.data(), x, y.data(), M, N);
    std::printf("cuda_ran=%d\n", cuda ? 1 : 0);
    std::printf("test_nv ok (__dp4a host-ref)\n");
    return 0;
}
