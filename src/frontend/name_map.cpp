#include "rapidllm/frontend/weight_store.h"

#include <regex>
#include <sstream>

namespace rapidllm {

static bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(0, p.size()) == p;
}

bool is_visual(std::string_view name) {
    return name.find("visual") != std::string_view::npos ||
           name.find("vision") != std::string_view::npos ||
           starts_with(name, "model.visual") || starts_with(name, "visual.");
}

bool is_visual_or_mtp(std::string_view name) { return is_visual(name); }

static std::string map_mtp_rest(std::string rest) {
    if (rest == "fc.weight") return "mtp.fc";
    if (rest == "norm.weight") return "mtp.norm";
    if (rest == "pre_fc_norm_hidden.weight") return "mtp.pre_fc_norm_hidden";
    if (rest == "pre_fc_norm_embedding.weight") return "mtp.pre_fc_norm_embedding";
    std::smatch m;
    static const std::regex re(R"(layers\.(\d+)\.(.+))");
    static const std::regex re2(R"(blk\.(\d+)\.(.+))");
    static const std::regex re3(R"((\d+)\.(.+))");
    std::string idx, tail;
    if (std::regex_match(rest, m, re)) {
        idx = m[1];
        tail = m[2];
    } else if (std::regex_match(rest, m, re2)) {
        idx = m[1];
        tail = m[2];
    } else if (std::regex_match(rest, m, re3)) {
        idx = m[1];
        tail = m[2];
    } else
        return {};
    const std::string p = "mtp.layers[" + idx + "].";
    if (tail == "input_layernorm.weight" || tail == "attn_norm.weight") return p + "attn_norm";
    if (tail == "post_attention_layernorm.weight" || tail == "ffn_norm.weight") return p + "ffn_norm";
    if (tail == "self_attn.q_proj.weight" || tail == "attn_q.weight") return p + "attn.wq";
    if (tail == "self_attn.k_proj.weight" || tail == "attn_k.weight") return p + "attn.wk";
    if (tail == "self_attn.v_proj.weight" || tail == "attn_v.weight") return p + "attn.wv";
    if (tail == "self_attn.o_proj.weight" || tail == "attn_output.weight") return p + "attn.wo";
    if (tail == "self_attn.q_norm.weight" || tail == "attn_q_norm.weight") return p + "attn.q_norm";
    if (tail == "self_attn.k_norm.weight" || tail == "attn_k_norm.weight") return p + "attn.k_norm";
    if (tail == "mlp.gate_proj.weight" || tail == "ffn_gate.weight") return p + "mlp.gate";
    if (tail == "mlp.up_proj.weight" || tail == "ffn_up.weight") return p + "mlp.up";
    if (tail == "mlp.down_proj.weight" || tail == "ffn_down.weight") return p + "mlp.down";
    return {};
}

bool is_leftover_name(std::string_view n) {
    auto has = [&](const char* k) { return n.find(k) != std::string_view::npos; };
    return has("A_log") || has("a_log") || has("dt_bias") || has("ssm_dt") ||
           has("conv1d") || has("ssm_conv") || has("shortconv") ||
           has("q_norm") || has("k_norm") || has("attn_q_norm") || has("attn_k_norm") ||
           has("input_layernorm") || has("post_attention_layernorm") ||
           has("attn_norm") || has("ffn_norm") || has("output_norm") ||
           has(".norm.weight") || has("ssm_norm") ||
           has("in_proj_a") || has("in_proj_b") || has("in_proj_ba") ||
           has("ssm_a") || has("ssm_beta");
}

static std::string layer_ir(int i, const std::string& rest) {
    return "layers[" + std::to_string(i) + "]." + rest;
}

std::string map_hf_name(std::string_view src) {
    std::string s(src);
    const std::string pref1 = "model.language_model.";
    const std::string pref2 = "model.";
    if (starts_with(s, pref1)) s = s.substr(pref1.size());
    else if (starts_with(s, pref2) && s.find("language_model") == std::string::npos &&
             s.find("visual") == std::string::npos)
        s = s.substr(pref2.size());

    if (starts_with(std::string(src), "mtp.") || starts_with(s, "mtp.")) {
        std::string rest = starts_with(std::string(src), "mtp.") ? std::string(src).substr(4) : s.substr(4);
        return map_mtp_rest(rest);
    }
    if (s == "embed_tokens.weight" || src == "model.embed_tokens.weight" ||
        src == "model.language_model.embed_tokens.weight")
        return "embed";
    if (s == "lm_head.weight" || src == "lm_head.weight") return "lm_head";
    if (s == "norm.weight" || s == "language_model.norm.weight") return "final_norm";

    std::smatch m;
    static const std::regex re(R"(layers\.(\d+)\.(.+))");
    if (!std::regex_match(s, m, re)) return {};

    const int i = std::stoi(m[1].str());
    const std::string rest = m[2].str();
    if (rest == "input_layernorm.weight") return layer_ir(i, "attn_norm");
    if (rest == "post_attention_layernorm.weight") return layer_ir(i, "ffn_norm");
    if (rest == "self_attn.q_proj.weight") return layer_ir(i, "attn.wq");
    if (rest == "self_attn.k_proj.weight") return layer_ir(i, "attn.wk");
    if (rest == "self_attn.v_proj.weight") return layer_ir(i, "attn.wv");
    if (rest == "self_attn.o_proj.weight") return layer_ir(i, "attn.wo");
    if (rest == "self_attn.q_norm.weight") return layer_ir(i, "attn.q_norm");
    if (rest == "self_attn.k_norm.weight") return layer_ir(i, "attn.k_norm");
    if (rest == "mlp.gate_proj.weight") return layer_ir(i, "mlp.gate");
    if (rest == "mlp.up_proj.weight") return layer_ir(i, "mlp.up");
    if (rest == "mlp.down_proj.weight") return layer_ir(i, "mlp.down");
    if (rest == "linear_attn.in_proj_qkv.weight" || rest == "linear_attn.in_proj_qkvz.weight")
        return layer_ir(i, "delta.gemm.in_proj_qkv");
    if (rest == "linear_attn.in_proj_z.weight") return layer_ir(i, "delta.gemm.in_proj_z");
    if (rest == "linear_attn.out_proj.weight") return layer_ir(i, "delta.gemm.out_proj");
    if (rest == "linear_attn.in_proj_a.weight") return layer_ir(i, "delta.leftover.in_proj_a");
    if (rest == "linear_attn.in_proj_b.weight") return layer_ir(i, "delta.leftover.in_proj_b");
    if (rest == "linear_attn.in_proj_ba.weight") return layer_ir(i, "delta.leftover.in_proj_ba");
    if (rest == "linear_attn.A_log") return layer_ir(i, "delta.leftover.a_log");
    if (rest == "linear_attn.dt_bias") return layer_ir(i, "delta.leftover.dt_bias");
    if (rest == "linear_attn.conv1d.weight") return layer_ir(i, "delta.leftover.conv1d");
    if (rest == "linear_attn.norm.weight") return layer_ir(i, "delta.leftover.norm");
    return {};
}

std::string map_gguf_name(std::string_view src) {
    std::string s(src);
    if (starts_with(s, "nextn.") || starts_with(s, "mtp.")) {
        std::string rest = s.substr(s.find('.') + 1);
        auto ir = map_mtp_rest(rest);
        if (!ir.empty()) return ir;
    }
    if (s == "token_embd.weight") return "embed";
    if (s == "output.weight") return "lm_head";
    if (s == "output_norm.weight") return "final_norm";

    std::smatch m;
    static const std::regex re(R"(blk\.(\d+)\.(.+))");
    if (!std::regex_match(s, m, re)) return {};
    const int i = std::stoi(m[1].str());
    const std::string rest = m[2].str();
    if (rest == "attn_norm.weight") return layer_ir(i, "attn_norm");
    if (rest == "ffn_norm.weight" || rest == "post_attention_norm.weight") return layer_ir(i, "ffn_norm");
    if (rest == "attn_q.weight") return layer_ir(i, "attn.wq");
    if (rest == "attn_k.weight") return layer_ir(i, "attn.wk");
    if (rest == "attn_v.weight") return layer_ir(i, "attn.wv");
    if (rest == "attn_output.weight" || rest == "attn_o.weight") return layer_ir(i, "attn.wo");
    if (rest == "attn_q_norm.weight") return layer_ir(i, "attn.q_norm");
    if (rest == "attn_k_norm.weight") return layer_ir(i, "attn.k_norm");
    if (rest == "ffn_gate.weight") return layer_ir(i, "mlp.gate");
    if (rest == "ffn_up.weight") return layer_ir(i, "mlp.up");
    if (rest == "ffn_down.weight") return layer_ir(i, "mlp.down");
    if (rest == "ssm_in.weight" || rest == "in_proj_qkv.weight" || rest == "attn_qkv.weight")
        return layer_ir(i, "delta.gemm.in_proj_qkv");
    if (rest == "ssm_z.weight" || rest == "in_proj_z.weight" || rest == "attn_gate.weight")
        return layer_ir(i, "delta.gemm.in_proj_z");
    if (rest == "ssm_out.weight") return layer_ir(i, "delta.gemm.out_proj");
    if (rest == "ssm_a.weight" || rest == "in_proj_a.weight" || rest == "ssm_alpha.weight")
        return layer_ir(i, "delta.leftover.in_proj_a");
    if (rest == "ssm_beta.weight" || rest == "in_proj_b.weight")
        return layer_ir(i, "delta.leftover.in_proj_b");
    if (rest == "ssm_a_log" || rest == "A_log" || rest == "ssm_a")
        return layer_ir(i, "delta.leftover.a_log");
    if (rest == "ssm_dt" || rest == "dt_bias" || rest == "ssm_dt.bias")
        return layer_ir(i, "delta.leftover.dt_bias");
    if (rest == "ssm_conv1d.weight" || rest == "shortconv.weight" || rest == "ssm_conv1d")
        return layer_ir(i, "delta.leftover.conv1d");
    if (rest == "ssm_norm.weight" || rest == "ssm_norm") return layer_ir(i, "delta.leftover.norm");
    return {};
}

} // namespace rapidllm
