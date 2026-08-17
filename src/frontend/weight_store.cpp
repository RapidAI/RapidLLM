#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rapidllm {
namespace {

bool should_materialize_f32(const TensorDesc& t) {
    if (t.quant != QuantKind::F16 && t.quant != QuantKind::BF16) return false;
    const std::string& n = t.ir_name;
    if (n.find("in_proj") != std::string::npos) return false;
    if (n.find("gemm") != std::string::npos) return false;
    if (n.find("mlp.") != std::string::npos) return false;
    if (n.find("attn.w") != std::string::npos) return false;
    if (n == "embed" || n == "lm_head") return false;
    if (n.find("norm") != std::string::npos) return true;
    if (n.find("a_log") != std::string::npos || n.find("dt_bias") != std::string::npos) return true;
    if (n.find("conv") != std::string::npos) return true;
    return t.ndim == 1;
}

// RapidLLM qwen3_rmsnorm applies (1+gamma). HF checkpoints store (w-1) after
// BF16 materialize or already match that convention. GGUF stores the raw
// scale (~1), so 1+w doubled the first RMS (xn0 l2 143 vs official 72).
bool is_qwen3_rms_gamma(const std::string& n) {
    if (n.find("leftover.norm") != std::string::npos) return false;
    if (n.find("attn_norm") != std::string::npos || n.find("ffn_norm") != std::string::npos) return true;
    if (n == "final_norm" || n == "mtp.norm") return true;
    if (n.find("pre_fc_norm") != std::string::npos) return true;
    if (n.find("q_norm") != std::string::npos || n.find("k_norm") != std::string::npos) return true;
    return false;
}

void gguf_shift_rms_gamma(TensorDesc& t) {
    if (t.quant != QuantKind::F32 || t.data.size() < 4) return;
    if (!is_qwen3_rms_gamma(t.ir_name)) return;
    float* p = reinterpret_cast<float*>(t.data.data());
    const size_t n = t.data.size() / sizeof(float);
    double mean = 0;
    for (size_t i = 0; i < n; ++i) mean += p[i];
    mean /= static_cast<double>(n);
    // Raw GGUF scale is ~1; already-shifted HF-style is ~0. Tiny fixtures vary.
    if (mean < 0.5) return;
    for (size_t i = 0; i < n; ++i) p[i] -= 1.f;
}

// llama.cpp Qwen3.5 convert writes blk.N.ssm_a = -exp(A_log) (F32, nv).
// RapidLLM kernels still do g = -exp(A_log) * softplus(a+dt). Invert so HF
// and GGUF share the same log-domain leftover. Tiny fixtures use ssm_a_log
// with raw A_log and must not go through this path.
bool src_is_llama_ssm_a(const std::string& src) {
    const size_t n = src.size();
    if (n < 5 || src.compare(n - 5, 5, "ssm_a") != 0) return false;
    if (n >= 7 && src.compare(n - 7, 7, "ssm_a_log") == 0) return false;
    if (n >= 13 && src.compare(n - 13, 13, "ssm_a.weight") == 0) return false;
    return true;
}

void gguf_unexp_ssm_a(TensorDesc& t) {
    if (t.quant != QuantKind::F32 || t.data.size() < 4) return;
    if (t.ir_name.find("a_log") == std::string::npos) return;
    if (!src_is_llama_ssm_a(t.src_name)) return;
    float* p = reinterpret_cast<float*>(t.data.data());
    const size_t n = t.data.size() / sizeof(float);
    // llama.cpp always stores -exp(A_log), including heads with A_log>=0
    // (value <= -1). Invert every element; do not abort the tensor.
    const float in0 = p[0];
    int nneg = 0;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] < 0.f) ++nneg;
        p[i] = std::log(std::fmax(-p[i], 1e-20f));
    }
    static int n_done = 0;
    ++n_done;
    if (t.src_name == "blk.0.ssm_a" || n_done == 1) {
        std::fprintf(stderr, "gguf_ssm_a_unexp src=%s n=%zu nneg=%d in0=%.6f out0=%.6f out1=%.6f done=%d\n",
                     t.src_name.c_str(), n, nneg, in0, p[0], n > 1 ? p[1] : 0.f, n_done);
    }
}

void materialize_f32(TensorDesc& t) {
    const size_t n = t.data.size() / 2;
    std::vector<float> out(n);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(t.data.data());
    if (t.quant == QuantKind::F16) {
        for (size_t i = 0; i < n; ++i) out[i] = ops::fp16_to_f32(h[i]);
    } else {
        for (size_t i = 0; i < n; ++i) {
            uint32_t u = static_cast<uint32_t>(h[i]) << 16;
            std::memcpy(&out[i], &u, 4);
        }
    }
    t.data.resize(n * sizeof(float));
    std::memcpy(t.data.data(), out.data(), n * sizeof(float));
    t.quant = QuantKind::F32;
    t.nbytes = n * sizeof(float);
}

} // namespace

std::unique_ptr<IWeightLoader> make_hf_loader();
std::unique_ptr<IWeightLoader> make_gguf_loader();

const TensorDesc* TensorTable::find(std::string_view ir_name) const {
    auto it = tensors.find(std::string(ir_name));
    return it == tensors.end() ? nullptr : &it->second;
}

TensorDesc* TensorTable::find_mut(std::string_view ir_name) {
    auto it = tensors.find(std::string(ir_name));
    return it == tensors.end() ? nullptr : &it->second;
}

uint64_t TensorTable::total_nbytes() const {
    uint64_t n = 0;
    for (const auto& [_, t] : tensors) n += t.nbytes;
    return n;
}

std::unique_ptr<IWeightLoader> make_loader(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        if (std::filesystem::exists(path / "config.json")) return make_hf_loader();
        throw LoadError("directory is not an HF model (missing config.json): " + path.string());
    }
    const auto ext = path.extension().string();
    std::string low = ext;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (low == ".gguf") return make_gguf_loader();
    throw LoadError("unknown model path (need HF dir or *.gguf): " + path.string());
}

WeightStore WeightStore::open(const std::filesystem::path& path, Device&, const LoadOptions& opt) {
    auto loader = make_loader(path);
    WeightStore s;
    s.table_ = loader->load(path, opt);
    for (auto& [_, t] : s.table_.tensors) {
        if (should_materialize_f32(t)) materialize_f32(t);
    }
    if (s.table_.source == SourceKind::GgufFile) {
        for (auto& [_, t] : s.table_.tensors) {
            gguf_shift_rms_gamma(t);
            gguf_unexp_ssm_a(t);
            if (src_is_llama_ssm_a(t.src_name) && t.ir_name.find("a_log") != std::string::npos)
                s.table_.model.gdn_v_tiled = true;
        }
        if (s.table_.model.gdn_v_tiled)
            std::fprintf(stderr, "gguf_gdn_v_tiled=1 (llama.cpp ssm_a V-head order)\n");
    }
    return s;
}

} // namespace rapidllm
