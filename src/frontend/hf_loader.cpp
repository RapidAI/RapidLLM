#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/kernels/ops.h"

#include "frontend/json_mini.h"
#include "frontend/safetensors.h"

#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace rapidllm {
namespace {

std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw LoadError("cannot open " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

QuantKind dtype_to_quant(const std::string& dt) {
    if (dt == "F32") return QuantKind::F32;
    if (dt == "F16") return QuantKind::F16;
    if (dt == "BF16") return QuantKind::BF16;
    if (dt == "F8_E4M3") return QuantKind::FP8_E4M3_B128;
    return QuantKind::F32;
}

void fill_from_text_config(ModelDesc& m, const Json& tc) {
    if (const Json* v = tc.get("hidden_size")) m.hidden = v->as_int();
    if (const Json* v = tc.get("num_hidden_layers")) m.n_layers = v->as_int();
    if (const Json* v = tc.get("intermediate_size")) m.intermediate = v->as_int();
    if (const Json* v = tc.get("vocab_size")) m.vocab = v->as_int();
    if (const Json* v = tc.get("max_position_embeddings")) m.max_pos = v->as_int();
    if (const Json* v = tc.get("rms_norm_eps")) m.rms_eps = static_cast<float>(v->as_num());
    if (const Json* v = tc.get("tie_word_embeddings")) m.tie_embed = v->is_bool() && v->as_bool();
    if (const Json* v = tc.get("mtp_num_hidden_layers")) m.mtp_n_layers = v->as_int();
    if (const Json* v = tc.get("mtp_use_dedicated_embeddings"))
        m.mtp_dedicated_embeddings = v->is_bool() && v->as_bool();
    if (const Json* v = tc.get("rope_theta")) m.rope.theta = static_cast<float>(v->as_num());
    if (const Json* rp = tc.get("rope_parameters"); rp && rp->is_obj()) {
        if (const Json* th = rp->get("rope_theta")) m.rope.theta = static_cast<float>(th->as_num());
        if (const Json* pf = rp->get("partial_rotary_factor"))
            m.rope.partial_factor = static_cast<float>(pf->as_num());
    }
    if (const Json* v = tc.get("partial_rotary_factor"))
        m.rope.partial_factor = static_cast<float>(v->as_num());

    DeltaNetDesc dproto;
    if (const Json* v = tc.get("linear_num_key_heads")) dproto.n_k_heads = v->as_int();
    if (const Json* v = tc.get("linear_num_value_heads")) dproto.n_v_heads = v->as_int();
    if (const Json* v = tc.get("linear_key_head_dim")) dproto.k_dim = v->as_int();
    if (const Json* v = tc.get("linear_value_head_dim")) dproto.v_dim = v->as_int();
    if (const Json* v = tc.get("linear_conv_kernel_dim")) dproto.conv_k = v->as_int();

    GatedAttnDesc aproto;
    if (const Json* v = tc.get("num_attention_heads")) aproto.n_q = v->as_int();
    if (const Json* v = tc.get("num_key_value_heads")) aproto.n_kv = v->as_int();
    if (const Json* v = tc.get("head_dim")) aproto.head_dim = v->as_int();

    std::vector<std::string> types;
    if (const Json* lt = tc.get("layer_types"); lt && lt->is_arr()) {
        for (const Json& e : lt->as_arr()) types.push_back(e.as_str());
    } else {
        throw LoadError("HF config missing text_config.layer_types");
    }

    m.embed = EmbeddingDesc{m.vocab, m.hidden, QuantKind::BF16, "embed"};
    m.lm_head = LinearDesc{m.vocab, m.hidden, false, QuantKind::BF16, 128, 128, "lm_head", {}};
    m.layers.resize(types.size());
    m.n_layers = static_cast<int>(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
        LayerDesc& L = m.layers[i];
        L.kind = layer_kind_from_type(types[i]);
        L.rms_eps = m.rms_eps;
        L.delta = dproto;
        L.attn = aproto;
        L.mlp.gate = LinearDesc{m.intermediate, m.hidden, false, QuantKind::F32, 128, 128, {}, {}};
        L.mlp.up = LinearDesc{m.intermediate, m.hidden, false, QuantKind::F32, 128, 128, {}, {}};
        L.mlp.down = LinearDesc{m.hidden, m.intermediate, false, QuantKind::F32, 128, 128, {}, {}};
    }
}

class HfLoader final : public IWeightLoader {
public:
    TensorTable load(const std::filesystem::path& path, const LoadOptions& opt) override {
        const auto cfg_path = path / "config.json";
        Json root = parse_json(slurp(cfg_path));
        const Json* tc = root.get("text_config");
        if (!tc) tc = &root;

        TensorTable table;
        table.source = SourceKind::HfFp8Dir;
        fill_from_text_config(table.model, *tc);
        table.model.language_only = opt.language_only;
        if (opt.max_layers > 0 && opt.max_layers < table.model.n_layers) {
            table.model.n_layers = opt.max_layers;
            table.model.layers.resize(static_cast<size_t>(opt.max_layers));
        }

        std::vector<std::filesystem::path> shards;
        const auto index_path = path / "model.safetensors.index.json";
        if (std::filesystem::exists(index_path)) {
            Json idx = parse_json(slurp(index_path));
            const Json* wm = idx.get("weight_map");
            if (!wm || !wm->is_obj()) throw LoadError("bad weight_map");
            std::map<std::string, int> seen;
            for (const auto& [_, file] : wm->as_obj()) {
                const std::string fn = file.as_str();
                if (!seen.count(fn)) {
                    seen[fn] = 1;
                    shards.push_back(path / fn);
                }
            }
        } else {
            const auto single = path / "model.safetensors";
            if (!std::filesystem::exists(single)) throw LoadError("no safetensors in " + path.string());
            shards.push_back(single);
        }

        std::unordered_map<std::string, STTensor> all;
        for (const auto& sh : shards) {
            SafetensorsFile f = read_safetensors(sh);
            for (auto& [n, t] : f.tensors) all.emplace(n, std::move(t));
        }

        int n_delta = 0, n_attn = 0;
        for (const LayerDesc& L : table.model.layers) {
            if (L.kind == LayerKind::GatedDeltaNet) ++n_delta;
            if (L.kind == LayerKind::GatedAttn) ++n_attn;
        }
        if (n_delta == 0) throw LoadError("HF hybrid missing DeltaNet layers");

        for (auto& [name, st] : all) {
            if (is_visual(name)) continue;
            std::string ir = map_hf_name(name);
            if (ir.empty()) {
                if (name.find("weight_scale_inv") != std::string::npos) continue;
                continue;
            }
            if (opt.max_layers > 0) {
                // skip layers beyond slice
                const auto pos = ir.find("layers[");
                if (pos == 0) {
                    const int li = std::stoi(ir.substr(7));
                    if (li >= opt.max_layers) continue;
                }
            }
            TensorDesc td;
            td.ir_name = ir;
            td.src_name = name;
            td.quant = dtype_to_quant(st.dtype);
            td.ndim = static_cast<int>(st.shape.size());
            for (int d = 0; d < td.ndim && d < 4; ++d) td.shape[d] = st.shape[static_cast<size_t>(d)];
            td.nbytes = st.bytes.size();
            td.data = std::move(st.bytes);

            if (st.dtype == "F8_E4M3") {
                const std::string scale_src = name + "_scale_inv";
                const std::string scale_src2 = [&] {
                    std::string s = name;
                    const auto w = s.rfind(".weight");
                    if (w != std::string::npos) s.replace(w, 7, ".weight_scale_inv");
                    return s;
                }();
                auto it = all.find(scale_src2);
                if (it == all.end()) it = all.find(name + ".weight_scale_inv");
                if (it == all.end()) {
                    throw LoadError("FP8 tensor missing weight_scale_inv: " + name);
                }
                td.scale_name = scale_src2;
                const std::string& sdt = it->second.dtype;
                if (sdt == "BF16") {
                    const size_t ns = it->second.bytes.size() / 2;
                    td.scale.resize(ns);
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(it->second.bytes.data());
                    for (size_t i = 0; i < ns; ++i) {
                        const uint32_t u = static_cast<uint32_t>(h[i]) << 16;
                        std::memcpy(&td.scale[i], &u, 4);
                    }
                } else if (sdt == "F16") {
                    const size_t ns = it->second.bytes.size() / 2;
                    td.scale.resize(ns);
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(it->second.bytes.data());
                    for (size_t i = 0; i < ns; ++i) td.scale[i] = ops::fp16_to_f32(h[i]);
                } else {
                    td.scale.resize(it->second.bytes.size() / 4);
                    std::memcpy(td.scale.data(), it->second.bytes.data(), it->second.bytes.size());
                }
                td.quant = QuantKind::FP8_E4M3_B128;
            }

            if (opt.reject_quantized_leftover && is_leftover_name(ir)) {
                if (td.quant == QuantKind::Q4_K || td.quant == QuantKind::Q5_K ||
                    td.quant == QuantKind::Q6_K || td.quant == QuantKind::Q8_0) {
                    throw LoadError("leftover quantized: " + name);
                }
            }
            table.tensors.emplace(ir, std::move(td));
        }

        auto need = [&](const std::string& ir) {
            if (!table.find(ir)) throw LoadError("HF missing required tensor " + ir);
        };
        need("embed");
        need("lm_head");
        need("final_norm");
        table.model.has_mtp = table.find("mtp.fc") != nullptr && table.find("mtp.norm") != nullptr;
        for (int i = 0; i < table.model.n_layers; ++i) {
            if (table.model.layers[static_cast<size_t>(i)].kind == LayerKind::GatedDeltaNet) {
                need("layers[" + std::to_string(i) + "].delta.leftover.a_log");
                need("layers[" + std::to_string(i) + "].delta.leftover.conv1d");
            }
        }
        return table;
    }
};

} // namespace

std::unique_ptr<IWeightLoader> make_hf_loader() { return std::make_unique<HfLoader>(); }

} // namespace rapidllm
