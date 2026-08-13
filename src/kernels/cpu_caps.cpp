#include "rapidllm/kernels/cpu_caps.h"
#include "rapidllm/kernels/ops.h"
#include "rapidllm/runtime/thread_pool.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace rapidllm::ops {
namespace {

void cpuid_ex(int leaf, int sub, uint32_t out[4]) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, leaf, sub);
    out[0] = static_cast<uint32_t>(regs[0]);
    out[1] = static_cast<uint32_t>(regs[1]);
    out[2] = static_cast<uint32_t>(regs[2]);
    out[3] = static_cast<uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
#endif
}

uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}

CpuCaps detect() {
    CpuCaps c;
    uint32_t r[4]{};
    cpuid_ex(0, 0, r);
    const uint32_t max_leaf = r[0];
    if (max_leaf < 1) return c;
    cpuid_ex(1, 0, r);
    c.sse2 = (r[3] & (1u << 26)) != 0;
    const bool osxsave = (r[2] & (1u << 27)) != 0;
    c.avx = (r[2] & (1u << 28)) != 0;
    c.fma = (r[2] & (1u << 12)) != 0;
    if (!osxsave || !c.avx) return c;
    const uint64_t xcr0 = xgetbv0();
    const bool ymm = (xcr0 & 0x6ull) == 0x6ull; // XMM + YMM
    if (!ymm) return c;
    if (max_leaf >= 7) {
        uint32_t e7[4]{};
        cpuid_ex(7, 0, e7);
        c.avx2 = (e7[1] & (1u << 5)) != 0;
        const bool zmm_os = (xcr0 & 0xE0ull) == 0xE0ull; // opmask + ZMM_hi256 + Hi16_ZMM
        if (zmm_os) {
            c.avx512f = (e7[1] & (1u << 16)) != 0;
            c.avx512dq = (e7[1] & (1u << 17)) != 0;
            c.avx512bw = (e7[1] & (1u << 30)) != 0;
            c.avx512vl = (e7[1] & (1u << 31)) != 0;
        }
    }
    return c;
}

} // namespace

const CpuCaps& cpu_caps() {
    static const CpuCaps caps = detect();
    return caps;
}

SimdKind active_simd() {
    const CpuCaps& c = cpu_caps();
    SimdKind want = SimdKind::Scalar;
    if (c.avx2 && c.fma) want = SimdKind::Avx2;
    if (c.avx512f && c.avx512dq && c.avx512bw && c.avx512vl) want = SimdKind::Avx512;

    const char* env = std::getenv("RAPIDLLM_SIMD");
    if (env && *env) {
        std::string s(env);
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (s == "scalar") return SimdKind::Scalar;
        if (s == "avx2") return (c.avx2 && c.fma) ? SimdKind::Avx2 : SimdKind::Scalar;
        if (s == "avx512" || s == "avx512f")
            return (c.avx512f && c.avx512dq) ? SimdKind::Avx512 : want;
    }
    return want;
}

const char* simd_isa_name() {
    switch (active_simd()) {
    case SimdKind::Avx512:
        return "avx512f+dq+bw+vl (runtime)";
    case SimdKind::Avx2:
        return "avx2+fma (runtime)";
    default:
        return "scalar (runtime)";
    }
}

static void gemv_one(SimdKind k, const float* W, const float* x, float* y, int m, int n) {
    switch (k) {
    case SimdKind::Avx512:
        gemv_f32_avx512(W, x, y, m, n);
        return;
    case SimdKind::Avx2:
        gemv_f32_avx2(W, x, y, m, n);
        return;
    default:
        gemv_f32_scalar(W, x, y, m, n);
        return;
    }
}

static void parallel_rows(int m, int n, const std::function<void(int, int)>& fn) {
    const long long work = static_cast<long long>(m) * n;
    if (work < 2000000LL || WorkerPool::instance().threads() <= 1) {
        fn(0, m);
        return;
    }
    WorkerPool::instance().parallel_for(m, fn);
}

void gemv_q8_0(const uint8_t* packed, const float* x, float* y, int m, int n) {
    const SimdKind k = active_simd();
    const int nb = n / 32;
    const size_t bsz = 34;
    parallel_rows(m, n, [&](int b, int e) {
        const uint8_t* Wp = packed + static_cast<size_t>(b) * nb * bsz;
        switch (k) {
        case SimdKind::Avx512:
            gemv_q8_0_avx512(Wp, x, y + b, e - b, n);
            break;
        case SimdKind::Avx2:
            gemv_q8_0_avx2(Wp, x, y + b, e - b, n);
            break;
        default:
            gemv_q8_0_scalar(Wp, x, y + b, e - b, n);
            break;
        }
    });
}

void gemv_f16(const uint16_t* W, const float* x, float* y, int m, int n) {
    const SimdKind k = active_simd();
    parallel_rows(m, n, [&](int b, int e) {
        const uint16_t* Wp = W + static_cast<size_t>(b) * n;
        switch (k) {
        case SimdKind::Avx512:
            gemv_f16_avx512(Wp, x, y + b, e - b, n);
            break;
        case SimdKind::Avx2:
            gemv_f16_avx2(Wp, x, y + b, e - b, n);
            break;
        default:
            gemv_f16_scalar(Wp, x, y + b, e - b, n);
            break;
        }
    });
}

void gemv_bf16(const uint16_t* W, const float* x, float* y, int m, int n) {
    const SimdKind k = active_simd();
    parallel_rows(m, n, [&](int b, int e) {
        const uint16_t* Wp = W + static_cast<size_t>(b) * n;
        switch (k) {
        case SimdKind::Avx512:
            gemv_bf16_avx512(Wp, x, y + b, e - b, n);
            break;
        case SimdKind::Avx2:
            gemv_bf16_avx2(Wp, x, y + b, e - b, n);
            break;
        default: {
            for (int i = 0; i < e - b; ++i) {
                const uint16_t* row = Wp + static_cast<size_t>(i) * n;
                float acc = 0.f;
                for (int j = 0; j < n; ++j) {
                    uint32_t u = static_cast<uint32_t>(row[j]) << 16;
                    float w;
                    std::memcpy(&w, &u, 4);
                    acc += w * x[j];
                }
                y[b + i] = acc;
            }
            break;
        }
        }
    });
}

void gemv_q4_k(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr size_t bsz = 144;
    const int nb = n / 256;
    parallel_rows(m, n, [&](int b, int e) {
        gemv_q4_k_scalar(packed + static_cast<size_t>(b) * nb * bsz, x, y + b, e - b, n);
    });
}

void gemv_q5_k(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr size_t bsz = 176;
    const int nb = n / 256;
    parallel_rows(m, n, [&](int b, int e) {
        gemv_q5_k_scalar(packed + static_cast<size_t>(b) * nb * bsz, x, y + b, e - b, n);
    });
}

void gemv_q6_k(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr size_t bsz = 210;
    const int nb = n / 256;
    parallel_rows(m, n, [&](int b, int e) {
        gemv_q6_k_scalar(packed + static_cast<size_t>(b) * nb * bsz, x, y + b, e - b, n);
    });
}

void gemv_f32_simd(const float* W, const float* x, float* y, int m, int n) {
    const SimdKind k = active_simd();
    const long long work = static_cast<long long>(m) * n;
    if (work < 2000000LL || WorkerPool::instance().threads() <= 1) {
        gemv_one(k, W, x, y, m, n);
        return;
    }
    WorkerPool::instance().parallel_for(m, [&](int b, int e) {
        gemv_one(k, W + static_cast<size_t>(b) * n, x, y + b, e - b, n);
    });
}

void delta_recurrent_step_simd(float* S, const float* q, const float* k, const float* v,
                               const float* beta, const float* g_log, float* o, int n_v, int dk,
                               int dv, float eps_l2) {
    const SimdKind kind = active_simd();
    auto one = [&](int h0, int h1) {
        const int nh = h1 - h0;
        if (nh <= 0) return;
        float* Sh = S + static_cast<size_t>(h0) * dk * dv;
        const float* qh = q + static_cast<size_t>(h0) * dk;
        const float* kh = k + static_cast<size_t>(h0) * dk;
        const float* vh = v + static_cast<size_t>(h0) * dv;
        float* oh = o + static_cast<size_t>(h0) * dv;
        switch (kind) {
        case SimdKind::Avx512:
            delta_recurrent_step_avx512(Sh, qh, kh, vh, beta + h0, g_log + h0, oh, nh, dk, dv, eps_l2);
            break;
        case SimdKind::Avx2:
            delta_recurrent_step_avx2(Sh, qh, kh, vh, beta + h0, g_log + h0, oh, nh, dk, dv, eps_l2);
            break;
        default:
            delta_recurrent_step_scalar(Sh, qh, kh, vh, beta + h0, g_log + h0, oh, nh, dk, dv, eps_l2);
            break;
        }
    };
    if (n_v >= 8 && WorkerPool::instance().threads() > 1) {
        WorkerPool::instance().parallel_for(n_v, one);
        return;
    }
    one(0, n_v);
}

} // namespace rapidllm::ops
