#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <cctype>
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
    return s;
}

} // namespace rapidllm
