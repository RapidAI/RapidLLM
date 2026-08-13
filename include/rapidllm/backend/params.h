#pragma once

namespace rapidllm {

struct RmsNormParams { int n; float eps; };
struct GemvFp8Params { int m, n; int block = 128; int act_dynamic = 0; };
struct GemvKQuantParams { int m, n; int type; };
struct GemvInt4Params { int m, n, group = 32; };
struct LmHeadParams { int hidden, vocab; int tile = 4096; int quant = 0; };
struct RopeParams {
    int n_q, n_kv, head_dim, rotary_dim;
    int pos;
    float theta;
    int mrope_section[3];
    int interleaved;
};
struct AttnDecodeParams { int n_q, n_kv, head_dim, seq, pos; int kv_dtype; float rms_eps; };
struct AttnPrefillParams { int n_q, n_kv, head_dim, seq; int br, bc; };
struct Conv1dParams { int dim, k, silu; };
struct DeltaRecurrentParams { int n_v, dk, dv; float eps_l2; };
struct DeltaChunkParams { int n_v, dk, dv, seq, chunk; float eps_l2; };
struct SwiGLUParams { int hidden, intermediate; };

} // namespace rapidllm
