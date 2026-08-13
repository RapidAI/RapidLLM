#include "rapidllm/ir/model_desc.h"

namespace rapidllm {

LayerKind layer_kind_from_type(std::string_view type) {
    if (type == "linear_attention") return LayerKind::GatedDeltaNet;
    if (type == "full_attention") return LayerKind::GatedAttn;
    if (type == "dense_attention") return LayerKind::DenseAttn;
    throw std::runtime_error(std::string("unknown layer_types entry: ") + std::string(type));
}

int ModelDesc::mixer_slot(int i) const {
    if (i < 0 || i >= static_cast<int>(layers.size())) {
        throw std::out_of_range("mixer_slot");
    }
    const LayerKind k = layers[static_cast<size_t>(i)].kind;
    int slot = 0;
    for (int j = 0; j < i; ++j) {
        if (layers[static_cast<size_t>(j)].kind == k) ++slot;
    }
    return slot;
}

void apply_layer_types(ModelDesc& m, const std::vector<std::string>& types) {
    if (types.empty()) throw std::runtime_error("layer_types empty");
    m.n_layers = static_cast<int>(types.size());
    m.layers.assign(types.size(), LayerDesc{});
    for (size_t i = 0; i < types.size(); ++i) {
        m.layers[i].kind = layer_kind_from_type(types[i]);
        m.layers[i].rms_eps = m.rms_eps;
        m.layers[i].delta.n_k_heads = m.layers[0].delta.n_k_heads;
        m.layers[i].delta.n_v_heads = (i == 0) ? 48 : m.layers[0].delta.n_v_heads;
    }
}

static void fill_delta_shapes(DeltaNetDesc& d, int hidden) {
    const int qkv = d.n_k_heads * d.k_dim + d.n_k_heads * d.k_dim + d.n_v_heads * d.v_dim;
    const int z = d.n_v_heads * d.v_dim;
    d.gemm.in_proj_qkv = LinearDesc{qkv, hidden, false, QuantKind::F32, 128, 128, {}, {}};
    d.gemm.in_proj_z = LinearDesc{z, hidden, false, QuantKind::F32, 128, 128, {}, {}};
    d.gemm.out_proj = LinearDesc{hidden, z, false, QuantKind::F32, 128, 128, {}, {}};
    d.leftover.in_proj_a = LinearDesc{d.n_v_heads, hidden, false, QuantKind::F32, 128, 128, {}, {}};
    d.leftover.in_proj_b = LinearDesc{d.n_v_heads, hidden, false, QuantKind::F32, 128, 128, {}, {}};
}

ModelDesc make_qwen36_27b_desc() {
    ModelDesc m;
    m.arch = ArchKind::Qwen35Hybrid;
    m.vocab = 248320;
    m.hidden = 5120;
    m.n_layers = 64;
    m.intermediate = 17408;
    m.max_pos = 262144;
    m.tie_embed = false;
    m.rms_eps = 1e-6f;
    m.embed = EmbeddingDesc{248320, 5120, QuantKind::BF16, "embed"};
    m.lm_head = LinearDesc{248320, 5120, false, QuantKind::BF16, 128, 128, "lm_head", {}};
    m.layers.resize(64);
    for (int i = 0; i < 64; ++i) {
        LayerDesc& L = m.layers[static_cast<size_t>(i)];
        L.kind = ((i % 4) == 3) ? LayerKind::GatedAttn : LayerKind::GatedDeltaNet;
        L.rms_eps = 1e-6f;
        L.delta = DeltaNetDesc{};
        fill_delta_shapes(L.delta, 5120);
        L.attn = GatedAttnDesc{};
        L.attn.wq = LinearDesc{24 * 256 * 2, 5120, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.attn.wk = LinearDesc{4 * 256, 5120, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.attn.wv = LinearDesc{4 * 256, 5120, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.attn.wo = LinearDesc{5120, 24 * 256, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.mlp.gate = LinearDesc{17408, 5120, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.mlp.up = LinearDesc{17408, 5120, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
        L.mlp.down = LinearDesc{5120, 17408, false, QuantKind::FP8_E4M3_B128, 128, 128, {}, {}};
    }
    return m;
}

ModelDesc make_tiny_hybrid_desc() {
    ModelDesc m;
    m.arch = ArchKind::Qwen35Hybrid;
    m.vocab = 48;
    m.hidden = 32;
    m.n_layers = 8;
    m.intermediate = 64;
    m.max_pos = 32768;
    m.tie_embed = false;
    m.rms_eps = 1e-6f;
    m.rope.theta = 10000.f;
    m.rope.partial_factor = 0.25f;
    m.embed = EmbeddingDesc{48, 32, QuantKind::F32, "embed"};
    m.lm_head = LinearDesc{48, 32, false, QuantKind::F32, 128, 128, "lm_head", {}};
    m.layers.resize(8);
    for (int i = 0; i < 8; ++i) {
        LayerDesc& L = m.layers[static_cast<size_t>(i)];
        L.kind = ((i % 4) == 3) ? LayerKind::GatedAttn : LayerKind::GatedDeltaNet;
        L.rms_eps = 1e-6f;
        L.delta.n_k_heads = 2;
        L.delta.n_v_heads = 6;
        L.delta.k_dim = 8;
        L.delta.v_dim = 8;
        L.delta.conv_k = 4;
        fill_delta_shapes(L.delta, 32);
        L.attn.n_q = 4;
        L.attn.n_kv = 2;
        L.attn.head_dim = 8;
        L.attn.wq = LinearDesc{4 * 8 * 2, 32, false, QuantKind::F32, 128, 128, {}, {}};
        L.attn.wk = LinearDesc{2 * 8, 32, false, QuantKind::F32, 128, 128, {}, {}};
        L.attn.wv = LinearDesc{2 * 8, 32, false, QuantKind::F32, 128, 128, {}, {}};
        L.attn.wo = LinearDesc{32, 4 * 8, false, QuantKind::F32, 128, 128, {}, {}};
        L.mlp.gate = LinearDesc{64, 32, false, QuantKind::F32, 128, 128, {}, {}};
        L.mlp.up = LinearDesc{64, 32, false, QuantKind::F32, 128, 128, {}, {}};
        L.mlp.down = LinearDesc{32, 64, false, QuantKind::F32, 128, 128, {}, {}};
    }
    return m;
}

} // namespace rapidllm
