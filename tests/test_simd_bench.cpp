#include "rapidllm/kernels/cpu_caps.h"
#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    using clock = std::chrono::steady_clock;
    using namespace rapidllm::ops;
    const CpuCaps& c = cpu_caps();
    std::printf("caps sse2=%d avx=%d avx2=%d fma=%d avx512f=%d avx512dq=%d avx512bw=%d avx512vl=%d\n",
                c.sse2, c.avx, c.avx2, c.fma, c.avx512f, c.avx512dq, c.avx512bw, c.avx512vl);
    std::printf("active=%s kind=%d\n", simd_isa_name(), static_cast<int>(active_simd()));

    constexpr int M = 1024, N = 1024, REPS = 12;
    std::vector<float> W(static_cast<size_t>(M) * N), x(N), ys(M), y2(M), y5(M), yd(M);
    for (int i = 0; i < M * N; ++i) W[static_cast<size_t>(i)] = 0.001f * static_cast<float>((i * 13) % 97);
    for (int i = 0; i < N; ++i) x[i] = 0.02f * static_cast<float>((i % 11) - 5);

    auto bench = [&](auto fn, float* y) {
        fn(W.data(), x.data(), y, M, N);
        const auto t0 = clock::now();
        for (int r = 0; r < REPS; ++r) fn(W.data(), x.data(), y, M, N);
        return std::chrono::duration<double>(clock::now() - t0).count() / REPS;
    };
    const double ts = bench(gemv_f32_scalar, ys.data());
    const double t2 = c.avx2 ? bench(gemv_f32_avx2, y2.data()) : 1e9;
    const double t5 = (c.avx512f && c.avx512dq) ? bench(gemv_f32_avx512, y5.data()) : 1e9;
    const double td = bench(gemv_f32_simd, yd.data());
    const auto gbs = [&](double t) { return (2.0 * M * N * 4.0) / t / 1e9; };
    std::printf("gemv_scalar  %.3f ms %.2f GB/s\n", ts * 1e3, gbs(ts));
    if (c.avx2) std::printf("gemv_avx2    %.3f ms %.2f GB/s\n", t2 * 1e3, gbs(t2));
    if (c.avx512f) std::printf("gemv_avx512  %.3f ms %.2f GB/s\n", t5 * 1e3, gbs(t5));
    std::printf("gemv_dispatch %.3f ms %.2f GB/s (%s)\n", td * 1e3, gbs(td), simd_isa_name());

    const bool simd_win = (c.avx2 && t2 < ts) || (c.avx512f && t5 < ts) || td < ts;
    if (!simd_win) {
        std::fprintf(stderr, "no SIMD GEMV beat scalar\n");
        return 1;
    }
    const char* force = std::getenv("RAPIDLLM_SIMD");
    if (!force || !*force) {
        if (c.avx512f && c.avx512dq && active_simd() != SimdKind::Avx512) {
            std::fprintf(stderr, "AVX-512 present but not selected\n");
            return 1;
        }
        if (!c.avx512f && c.avx2 && c.fma && active_simd() != SimdKind::Avx2) {
            std::fprintf(stderr, "AVX2 present but not selected\n");
            return 1;
        }
    }
    // Q8_0 fused SIMD must match scalar dequant-dot.
    constexpr int MQ = 64, NQ = 256;
    std::vector<float> Wq(static_cast<size_t>(MQ) * NQ), xq(NQ), ysc(MQ), ysi(MQ);
    for (int i = 0; i < MQ * NQ; ++i) Wq[static_cast<size_t>(i)] = 0.02f * static_cast<float>((i * 7) % 51 - 25);
    for (int i = 0; i < NQ; ++i) xq[i] = 0.03f * static_cast<float>((i % 9) - 4);
    const int nb = NQ / 32;
    std::vector<uint8_t> packed(static_cast<size_t>(MQ) * nb * 34);
    for (int r = 0; r < MQ; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* src = Wq.data() + r * NQ + b * 32;
            float amax = 0.f;
            for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(src[k]));
            const float d = amax > 0.f ? amax / 127.f : 1.f;
            uint8_t* blk = packed.data() + (static_cast<size_t>(r) * nb + b) * 34;
            const uint16_t dh = f32_to_fp16(d);
            std::memcpy(blk, &dh, 2);
            for (int k = 0; k < 32; ++k) {
                int q = static_cast<int>(std::lround(src[k] / d));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                blk[2 + k] = static_cast<uint8_t>(static_cast<int8_t>(q));
            }
        }
    }
    gemv_q8_0_scalar(packed.data(), xq.data(), ysc.data(), MQ, NQ);
    gemv_q8_0(packed.data(), xq.data(), ysi.data(), MQ, NQ);
    float maxe = 0.f;
    for (int i = 0; i < MQ; ++i) maxe = std::max(maxe, std::fabs(ysc[i] - ysi[i]));
    if (maxe > 1e-4f) {
        std::fprintf(stderr, "q8_0 simd vs scalar maxerr=%g\n", maxe);
        return 1;
    }
    auto bench_q = [&](auto fn) {
        fn(packed.data(), xq.data(), ysi.data(), MQ, NQ);
        const auto t0 = clock::now();
        for (int r = 0; r < 40; ++r) fn(packed.data(), xq.data(), ysi.data(), MQ, NQ);
        return std::chrono::duration<double>(clock::now() - t0).count() / 40;
    };
    const double tqs = bench_q(gemv_q8_0_scalar);
    const double tqd = bench_q(gemv_q8_0);
    std::printf("q8_0_scalar  %.3f ms  q8_0_dispatch %.3f ms  maxerr=%g\n", tqs * 1e3, tqd * 1e3, maxe);

    std::printf("simd_bench ok dispatched=%s\n", simd_isa_name());
    return 0;
}
