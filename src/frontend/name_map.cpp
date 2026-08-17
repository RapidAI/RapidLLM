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

static std::string map_visual_name(std::string_view src) {
    std::string s(src);
    const char* prefs[] = {"model.visual.", "visual."};
    for (const char* p : prefs) {
        if (starts_with(s, p)) {
            s = s.substr(std::char_traits<char>::length(p));
            break;
        }
    }
    auto strip_w = [](std::string x) {
        if (x.size() > 7 && x.substr(x.size() - 7) == ".weight") x.resize(x.size() - 7);
        return x;
    };
    s = strip_w(s);
    if (s == "patch_embed.proj") return "visual.patch_embed";
    if (s == "patch_embed.proj.bias") return "visual.patch_embed_bias";
    if (s == "pos_embed" || s == "pos_embed.weight") return "visual.pos_embed";
    if (s == "merger.linear_fc1") return "visual.merger.fc1";
    if (s == "merger.linear_fc1.bias") return "visual.merger.fc1_bias";
    if (s == "merger.linear_fc2") return "visual.merger.fc2";
    if (s == "merger.linear_fc2.bias") return "visual.merger.fc2_bias";
    if (s == "merger.norm") return "visual.merger.norm";
    if (s == "merger.norm.bias") return "visual.merger.norm_bias";
    std::smatch m;
    static const std::regex re(R"(blocks\.(\d+)\.(.+))");
    if (!std::regex_match(s, m, re)) return {};
    const std::string p = "visual.blocks[" + m[1].str() + "].";
    const std::string r = m[2].str();
    if (r == "attn.qkv") return p + "attn.qkv";
    if (r == "attn.qkv.bias") return p + "attn.qkv_bias";
    if (r == "attn.proj") return p + "attn.proj";
    if (r == "attn.proj.bias") return p + "attn.proj_bias";
    if (r == "mlp.linear_fc1") return p + "mlp.fc1";
    if (r == "mlp.linear_fc1.bias") return p + "mlp.fc1_bias";
    if (r == "mlp.linear_fc2") return p + "mlp.fc2";
    if (r == "mlp.linear_fc2.bias") return p + "mlp.fc2_bias";
    if (r == "norm1") return p + "norm1";
    if (r == "norm1.bias") return p + "norm1_bias";
    if (r == "norm2") return p + "norm2";
    if (r == "norm2.bias") return p + "norm2_bias";
    return {};
}

std::string map_hf_name(std::string_view src) {
    if (is_visual(src)) return map_visual_name(src);
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
    // Jackrong / llama.cpp qwen35 MTP-GGUF: last block is the NextN layer.
    // blk.64.nextn.eh_proj / enorm / hnorm / shared_head_norm.
    {
        std::smatch nm;
        static const std::regex re_blk_nextn(R"(blk\.(\d+)\.nextn\.(.+))");
        if (std::regex_match(s, nm, re_blk_nextn)) {
            const std::string tail = nm[2].str();
            if (tail == "eh_proj.weight" || tail == "eh_proj") return "mtp.fc";
            if (tail == "enorm.weight" || tail == "enorm") return "mtp.pre_fc_norm_embedding";
            if (tail == "hnorm.weight" || tail == "hnorm") return "mtp.pre_fc_norm_hidden";
            if (tail == "shared_head_norm.weight" || tail == "shared_head_norm") return "mtp.norm";
            // blk.N.nextn.attn_q / ffn_* live on the NextN block in some GGUFs.
            auto ir = map_mtp_rest(std::string("0.") + tail);
            if (!ir.empty()) return ir;
        }
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
