#pragma once

namespace rapidllm::cuda_mtp {

// Isolated MTP decode attention. Float KV, short ctx. t_lo skips empty
// prefix slots (fork streams at pos>=1, cache[0] is never written).
void launch_mtp_attn_decode(float* q, const float* k, const float* v, const float* q_norm,
                            const float* k_norm, float* k_cache, float* v_cache, float* o,
                            const int* pos, int n_q, int n_kv, int hd, int rotary, float theta,
                            float eps, int ctx, int t_lo);

} // namespace rapidllm::cuda_mtp
