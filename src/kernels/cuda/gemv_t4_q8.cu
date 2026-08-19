// Isolated T=4 / T=12 Q8. One W pass, smem X tile.
// T=4 1-row + launch_bounds(512,2) tot ~45ms (plain 1-row was 48).
// T=12 tile=960 tot ~125ms. 2-row spilled (T=4 335ms, T=12 launch fail).
#include "rapidllm/kernels/gemv_t4_q8.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kXsPad = 33;
constexpr int kTile = 2048;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

__device__ __forceinline__ int xs_off(int b) { return b * kXsPad; }

__device__ __forceinline__ float q8_dot32(const int8_t* q, const float* x) {
    float s = 0.f;
    const int4 a = __ldcs(reinterpret_cast<const int4*>(q));
    const int4 b = __ldcs(reinterpret_cast<const int4*>(q + 16));
    const signed char* qa = reinterpret_cast<const signed char*>(&a);
    const signed char* qb = reinterpret_cast<const signed char*>(&b);
#pragma unroll
    for (int k = 0; k < 16; ++k) s += static_cast<float>(qa[k]) * x[k];
#pragma unroll
    for (int k = 0; k < 16; ++k) s += static_cast<float>(qb[k]) * x[16 + k];
    return s;
}

__global__ void __launch_bounds__(512, 2) gemm_q8_soa_t4_k(const int8_t* Q, const __half* scales, const float* X,
                                                           float* Y, int m, int n, int add) {
    extern __shared__ float xs[];
    constexpr int T = 4, tile = kTile;
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    const int tstride = (tile / 32) * kXsPad;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int8_t* rowq = row < m ? Q + static_cast<size_t>(row) * n : nullptr;
    const __half* rows = row < m ? scales + static_cast<size_t>(row) * nb : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < T * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[static_cast<size_t>(t) * tstride + (j >> 5) * kXsPad + (j & 31)] =
                X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        if (rowq) {
            const int bn = tn / 32;
            for (int b = lane; b < bn; b += 32) {
                const float d = __half2float(__ldcs(rows + t0 / 32 + b));
                const int8_t* q = rowq + t0 + b * 32;
                acc0 += d * q8_dot32(q, xs + xs_off(b));
                acc1 += d * q8_dot32(q, xs + tstride + xs_off(b));
                acc2 += d * q8_dot32(q, xs + 2 * tstride + xs_off(b));
                acc3 += d * q8_dot32(q, xs + 3 * tstride + xs_off(b));
            }
        }
        __syncthreads();
    }
    if (row < m) {
        const float a0 = warp_sum(acc0);
        const float a1 = warp_sum(acc1);
        const float a2 = warp_sum(acc2);
        const float a3 = warp_sum(acc3);
        if (lane == 0) {
            write_y(Y, row, a0, add);
            write_y(Y + m, row, a1, add);
            write_y(Y + 2 * m, row, a2, add);
            write_y(Y + 3 * m, row, a3, add);
        }
    }
}

// tile=960 → 12*30*33*4 = 47520 < 48 KiB. 6 tiles.
__global__ void gemm_q8_soa_t12_k(const int8_t* Q, const __half* scales, const float* X, float* Y, int m,
                                  int n, int add) {
    extern __shared__ float xs[];
    constexpr int T = 12, tile = 960;
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    const int tstride = (tile / 32) * kXsPad;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.f;
    const int8_t* rowq = row < m ? Q + static_cast<size_t>(row) * n : nullptr;
    const __half* rows = row < m ? scales + static_cast<size_t>(row) * nb : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < T * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[static_cast<size_t>(t) * tstride + (j >> 5) * kXsPad + (j & 31)] =
                X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        if (rowq) {
            const int bn = tn / 32;
            for (int b = lane; b < bn; b += 32) {
                const float d = __half2float(__ldcs(rows + t0 / 32 + b));
                const int8_t* q = rowq + t0 + b * 32;
#pragma unroll
                for (int t = 0; t < T; ++t) acc[t] += d * q8_dot32(q, xs + t * tstride + xs_off(b));
            }
        }
        __syncthreads();
    }
    if (row < m) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) write_y(Y + static_cast<size_t>(t) * m, row, a, add);
        }
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q8_f32_t4_1row(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n,
                           int add) {
    if (!Q || !scales || !X || !Y || m <= 0 || n <= 0 || (n % 32) != 0) return;
    const int warps = m >= 2048 ? 16 : 8;
    const int th = warps * 32;
    const int pb = (m + warps - 1) / warps;
    const size_t smem = static_cast<size_t>(4) * (kTile / 32) * kXsPad * sizeof(float);
    gemm_q8_soa_t4_k<<<pb, th, smem>>>(Q, scales, X, Y, m, n, add);
}

void launch_q8_f32_t4_1row_dual(const int8_t* Q1, const __half* S1, const int8_t* Q2, const __half* S2,
                                const float* X, float* Y1, float* Y2, int m, int n) {
    launch_q8_f32_t4_1row(Q1, S1, X, Y1, m, n, 0);
    if (Y2) launch_q8_f32_t4_1row(Q2, S2, X, Y2, m, n, 0);
}

void launch_q8_f32_t12(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n, int add) {
    if (!Q || !scales || !X || !Y || m <= 0 || n <= 0 || (n % 32) != 0) return;
    const int warps = m >= 2048 ? 16 : 8;
    const int th = warps * 32;
    const int pb = (m + warps - 1) / warps;
    constexpr int tile = 960;
    const size_t smem = static_cast<size_t>(12) * (tile / 32) * kXsPad * sizeof(float);
    gemm_q8_soa_t12_k<<<pb, th, smem>>>(Q, scales, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
