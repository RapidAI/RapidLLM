#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rapidllm {

enum class ArchKind { Qwen35Hybrid, Qwen3Dense, Qwen2Dense };
enum class LayerKind { GatedDeltaNet, GatedAttn, DenseAttn };
enum class ActKind { SiLU };
enum class RopeKind { None, PartialMrope, Yarn };
enum class QuantKind {
    F32,
    F16,
    BF16,
    FP8_E4M3_B128,
    PackedInt4,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    IQUnknown
};
enum class DType { F32, F16, BF16, F8_E4M3, I8, I32, Q4K, Q5K, Q6K, Q8_0 };

struct EmbeddingDesc {
    int vocab = 0, hidden = 0;
    QuantKind quant = QuantKind::BF16;
    std::string weight_name;
};

struct LinearDesc {
    int rows = 0, cols = 0;
    bool bias = false;
    QuantKind quant = QuantKind::FP8_E4M3_B128;
    int block_m = 128, block_n = 128;
    std::string weight_name;
    std::string scale_name;
};

struct RopeDesc {
    RopeKind kind = RopeKind::PartialMrope;
    float theta = 1e7f;
    float partial_factor = 0.25f;
    bool mrope_interleaved = true;
    int mrope_section[3] = {11, 11, 10};
    float yarn_factor = 1.f;
    int original_max_pos = 262144;
};

struct DeltaNetLargeGemm {
    LinearDesc in_proj_qkv;
    LinearDesc in_proj_z;
    LinearDesc out_proj;
};

struct DeltaNetLeftover {
    std::string a_log;
    std::string dt_bias;
    std::string conv1d;
    std::string norm;
    LinearDesc in_proj_a;
    LinearDesc in_proj_b;
    LinearDesc in_proj_ba;
};

struct DeltaNetDesc {
    int n_k_heads = 16, n_v_heads = 48;
    int k_dim = 128, v_dim = 128;
    int conv_k = 4;
    DeltaNetLargeGemm gemm;
    DeltaNetLeftover leftover;
};

struct GatedAttnDesc {
    int n_q = 24, n_kv = 4, head_dim = 256;
    bool qk_norm = true;
    bool output_gate = true;
    LinearDesc wq, wk, wv, wo;
};

struct MlpDesc {
    ActKind act = ActKind::SiLU;
    LinearDesc gate, up, down;
};

struct LayerDesc {
    LayerKind kind = LayerKind::GatedDeltaNet;
    float rms_eps = 1e-6f;
    DeltaNetDesc delta;
    GatedAttnDesc attn;
    MlpDesc mlp;
};

struct VisionDesc {
    bool present = false;
    int depth = 27;
    int hidden = 1152;
    int intermediate = 4304;
    int n_heads = 16;
    int in_channels = 3;
    int patch = 16;
    int temporal_patch = 2;
    int spatial_merge = 2;
    int out_hidden = 5120;
    int n_pos = 2304;
    int image_token_id = 248056;
    int video_token_id = 248057;
    int vision_start_id = 248053;
    int vision_end_id = 248054;
};

struct ModelDesc {
    ArchKind arch = ArchKind::Qwen35Hybrid;
    int vocab = 248320, hidden = 5120, n_layers = 64;
    int intermediate = 17408;
    int max_pos = 262144;
    bool tie_embed = false;
    bool language_only = true;
    float rms_eps = 1e-6f;
    RopeDesc rope;
    EmbeddingDesc embed;
    LinearDesc lm_head;
    std::vector<LayerDesc> layers;
    VisionDesc vision;
    int mtp_n_layers = 1;
    bool mtp_dedicated_embeddings = false;
    bool has_mtp = false;

    int mixer_slot(int i) const;
    LayerKind kind_of(int i) const { return layers.at(static_cast<size_t>(i)).kind; }
};

LayerKind layer_kind_from_type(std::string_view type);
ModelDesc make_qwen36_27b_desc();
ModelDesc make_tiny_hybrid_desc();
void apply_layer_types(ModelDesc& m, const std::vector<std::string>& types);

} // namespace rapidllm
