// FlashInfer single-request GQA prefill. Q is F32 NHD after our RMS+RoPE;
// K/V are the engine's F16 cache [kv_len, n_kv, hd]. Causal: Q is the last
// qo_len tokens of kv_len (chunked prefill).
#include "rapidllm/kernels/flashinfer_attn.h"

#if defined(RAPIDLLM_FLASHINFER)

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

#include <cuda_fp16.h>
#include <cstdio>

namespace {

__global__ void f32_to_f16_n(const float* x, __half* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void f16_to_f32_n(const __half* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

} // namespace

bool fi_available() { return true; }

bool fi_prefill_gqa(const float* q, const void* k_f16, const void* v_f16, float* o, int qo_len,
                    int kv_len, int n_q, int n_kv, int hd, float* scratch, size_t scratch_floats) {
    if (!q || !k_f16 || !v_f16 || !o || !scratch) return false;
    if (hd != 256 || qo_len <= 0 || kv_len < qo_len || n_q <= 0 || n_kv <= 0) return false;
    if (n_q % n_kv != 0) return false;
    // FlashInfer DISPATCH_GQA_GROUP_SIZE is 1/2/3/4/8. Official 27B is 24/4=6.
    const int g = n_q / n_kv;
    if (g != 1 && g != 2 && g != 3 && g != 4 && g != 8) return false;
    const size_t qn = static_cast<size_t>(qo_len) * n_q * hd;
    // q_f16 + o_f16 + split-KV tmp (up to 32 chunks)
    const size_t need = qn + qn + static_cast<size_t>(32) * n_q * hd;
    if (need > scratch_floats) return false;

    __half* qh = reinterpret_cast<__half*>(scratch);
    __half* oh = qh + qn;
    __half* tmp = oh + qn;
    const int n = static_cast<int>(qn);
    f32_to_f16_n<<<(n + 255) / 256, 256>>>(q, qh, n);

    using namespace flashinfer;
    using Params = SinglePrefillParams<half, half, half>;
    using Variant = DefaultAttention<false, false, false, false>;
    Params params(qh, reinterpret_cast<half*>(const_cast<void*>(k_f16)),
                  reinterpret_cast<half*>(const_cast<void*>(v_f16)), nullptr, oh, nullptr, nullptr,
                  static_cast<uint32_t>(n_q), static_cast<uint32_t>(n_kv),
                  static_cast<uint32_t>(qo_len), static_cast<uint32_t>(kv_len),
                  static_cast<uint32_t>(n_q * hd), static_cast<uint32_t>(hd),
                  static_cast<uint32_t>(n_kv * hd), static_cast<uint32_t>(hd), 256, -1, 0.f, 0.0625f,
                  1.f, 1e4f);
    try {
        const cudaError_t st = SinglePrefillWithKVCacheDispatched<
            256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal, Variant, Params>(
            params, tmp, cudaStreamPerThread);
        if (st != cudaSuccess) return false;
    } catch (...) {
        return false;
    }
    f16_to_f32_n<<<(n + 255) / 256, 256>>>(oh, o, n);
    return true;
}

#endif
