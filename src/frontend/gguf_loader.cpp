#include "rapidllm/frontend/weight_store.h"

#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

namespace rapidllm {
namespace {

enum GgufType : uint32_t {
    GGUF_U8 = 0,
    GGUF_I8 = 1,
    GGUF_U16 = 2,
    GGUF_I16 = 3,
    GGUF_U32 = 4,
    GGUF_I32 = 5,
    GGUF_F32 = 6,
    GGUF_BOOL = 7,
    GGUF_STR = 8,
    GGUF_ARR = 9,
    GGUF_U64 = 10,
    GGUF_I64 = 11,
    GGUF_F64 = 12
};

enum GgmlType : int {
    GGML_F32 = 0,
    GGML_F16 = 1,
    GGML_Q8_0 = 8,
    GGML_Q4_K = 12,
    GGML_Q5_K = 13,
    GGML_Q6_K = 14,
    GGML_BF16 = 30
};

struct Reader {
    std::ifstream in;
    explicit Reader(const std::filesystem::path& p) : in(p, std::ios::binary) {
        if (!in) throw LoadError("cannot open " + p.string());
    }
    template <class T>
    T rd() {
        T v{};
        in.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!in) throw LoadError("GGUF truncated");
        return v;
    }
    std::string rd_str() {
        const uint64_t n = rd<uint64_t>();
        if (n > 16ull * 1024 * 1024) throw LoadError("GGUF string too long");
        std::string s(static_cast<size_t>(n), '\0');
        if (n) in.read(s.data(), static_cast<std::streamsize>(n));
        if (!in) throw LoadError("GGUF truncated string");
        return s;
    }
};

void skip_value(Reader& r, uint32_t t);

void skip_array(Reader& r) {
    const uint32_t at = r.rd<uint32_t>();
    const uint64_t n = r.rd<uint64_t>();
    for (uint64_t i = 0; i < n; ++i) skip_value(r, at);
}

void skip_value(Reader& r, uint32_t t) {
    switch (t) {
    case GGUF_U8:
    case GGUF_I8:
    case GGUF_BOOL:
        r.rd<uint8_t>();
        break;
    case GGUF_U16:
    case GGUF_I16:
        r.rd<uint16_t>();
        break;
    case GGUF_U32:
    case GGUF_I32:
    case GGUF_F32:
        r.rd<uint32_t>();
        break;
    case GGUF_U64:
    case GGUF_I64:
    case GGUF_F64:
        r.rd<uint64_t>();
        break;
    case GGUF_STR:
        r.rd_str();
        break;
    case GGUF_ARR:
        skip_array(r);
        break;
    default:
        throw LoadError("GGUF unknown value type");
    }
}

QuantKind ggml_to_quant(int t) {
    switch (t) {
    case GGML_F32:
        return QuantKind::F32;
    case GGML_F16:
        return QuantKind::F16;
    case GGML_BF16:
        return QuantKind::BF16;
    case GGML_Q8_0:
        return QuantKind::Q8_0;
    case GGML_Q4_K:
        return QuantKind::Q4_K;
    case GGML_Q5_K:
        return QuantKind::Q5_K;
    case GGML_Q6_K:
        return QuantKind::Q6_K;
    default:
        return QuantKind::IQUnknown;
    }
}

bool leftover_quant_forbidden(QuantKind q) {
    return q == QuantKind::Q4_K || q == QuantKind::Q5_K || q == QuantKind::Q6_K ||
           q == QuantKind::Q8_0 || q == QuantKind::IQUnknown;
}

class GgufLoaderImpl final : public IWeightLoader {
public:
    TensorTable load(const std::filesystem::path& path, const LoadOptions& opt) override {
        Reader r(path);
        char magic[4];
        r.in.read(magic, 4);
        if (std::string(magic, 4) != "GGUF") throw LoadError("not a GGUF file");
        const uint32_t version = r.rd<uint32_t>();
        if (version < 2 || version > 3) throw LoadError("unsupported GGUF version");
        const uint64_t n_tensors = r.rd<uint64_t>();
        const uint64_t n_kv = r.rd<uint64_t>();

        auto rd_int = [&](uint32_t vt) -> int64_t {
            if (vt == GGUF_U32 || vt == GGUF_I32) return static_cast<int64_t>(r.rd<uint32_t>());
            if (vt == GGUF_U64 || vt == GGUF_I64) return static_cast<int64_t>(r.rd<uint64_t>());
            skip_value(r, vt);
            return -1;
        };
        std::string architecture;
        int64_t block_count = -1;
        int64_t n_embd = -1;
        int64_t n_vocab = -1;
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = r.rd_str();
            const uint32_t vt = r.rd<uint32_t>();
            if (key == "general.architecture" && vt == GGUF_STR) {
                architecture = r.rd_str();
            } else if (key.find("block_count") != std::string::npos &&
                       (vt == GGUF_U32 || vt == GGUF_I32 || vt == GGUF_U64 || vt == GGUF_I64)) {
                block_count = rd_int(vt);
            } else if (key.find("embedding_length") != std::string::npos &&
                       (vt == GGUF_U32 || vt == GGUF_I32 || vt == GGUF_U64 || vt == GGUF_I64)) {
                n_embd = rd_int(vt);
            } else if ((key.find("vocab_size") != std::string::npos || key == "tokenizer.ggml.model") &&
                       (vt == GGUF_U32 || vt == GGUF_I32 || vt == GGUF_U64 || vt == GGUF_I64)) {
                if (key.find("vocab_size") != std::string::npos) n_vocab = rd_int(vt);
                else skip_value(r, vt);
            } else {
                skip_value(r, vt);
            }
        }

        struct Meta {
            std::string name;
            std::vector<int64_t> shape;
            int ggml = 0;
            uint64_t offset = 0;
        };
        std::vector<Meta> metas;
        metas.reserve(static_cast<size_t>(n_tensors));
        for (uint64_t i = 0; i < n_tensors; ++i) {
            Meta m;
            m.name = r.rd_str();
            const uint32_t ndim = r.rd<uint32_t>();
            m.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) m.shape[d] = static_cast<int64_t>(r.rd<uint64_t>());
            // GGUF stores dims reversed vs typical row-major names; keep file order (last dim fastest)
            std::reverse(m.shape.begin(), m.shape.end());
            m.ggml = static_cast<int>(r.rd<uint32_t>());
            m.offset = r.rd<uint64_t>();
            metas.push_back(std::move(m));
        }

        const uint64_t alignment = 32;
        const uint64_t data_start = (static_cast<uint64_t>(r.in.tellg()) + alignment - 1) / alignment * alignment;

        TensorTable table;
        table.source = SourceKind::GgufFile;
        table.model = make_tiny_hybrid_desc(); // overwritten after probe

        int max_blk = -1;
        int n_delta = 0, n_attn = 0;
        std::vector<int> kind(256, -1);
        for (const Meta& m : metas) {
            if (is_visual(m.name)) continue;
            if (m.name.rfind("blk.", 0) == 0) {
                const int blk = std::stoi(m.name.substr(4));
                max_blk = std::max(max_blk, blk);
                if (m.name.find("ssm_") != std::string::npos || m.name.find("linear_attn") != std::string::npos ||
                    m.name.find("A_log") != std::string::npos || m.name.find("attn_qkv") != std::string::npos)
                    kind[static_cast<size_t>(blk)] = 0; // Gated DeltaNet (attn_qkv != attn_q)
                if (m.name.find("attn_q.weight") != std::string::npos ||
                    m.name.find("attn_q_norm") != std::string::npos)
                    kind[static_cast<size_t>(blk)] = 1; // Gated Attention
            }
        }
        if (max_blk < 0) throw LoadError("GGUF has no blk.* tensors; architecture=" + architecture);

        const int n_layers = max_blk + 1;
        for (int i = 0; i < n_layers; ++i) {
            if (kind[static_cast<size_t>(i)] == 0) ++n_delta;
            else if (kind[static_cast<size_t>(i)] == 1) ++n_attn;
        }
        if (n_delta == 0) {
            throw LoadError("GGUF missing DeltaNet tensors (A_log/ssm_*); architecture=" + architecture);
        }

        int64_t embed_vocab = -1, embed_hidden = -1;
        for (const Meta& m : metas) {
            if (m.name == "token_embd.weight" && m.shape.size() >= 2) {
                embed_vocab = m.shape[0];
                embed_hidden = m.shape[1];
            }
        }
        if (n_embd < 0 && embed_hidden > 0) n_embd = embed_hidden;
        if (n_vocab < 0 && embed_vocab > 0) n_vocab = embed_vocab;
        // Qwen3.5 / 3.6 / 3.8 27B share the same hybrid IR (hidden 5120, 64 layers).
        const bool family_27b = architecture.find("qwen35") != std::string::npos ||
                                architecture.find("qwen3_5") != std::string::npos ||
                                architecture.find("qwen36") != std::string::npos ||
                                architecture.find("qwen38") != std::string::npos ||
                                architecture.find("qwen3_8") != std::string::npos ||
                                architecture.find("qwen3.8") != std::string::npos;
        const bool is_27b = (n_layers >= 32 && (n_embd == 5120 || embed_hidden == 5120)) ||
                            (family_27b && n_layers >= 32);
        table.model = is_27b ? make_qwen36_27b_desc() : make_tiny_hybrid_desc();
        if (n_layers != table.model.n_layers) {
            table.model.n_layers = n_layers;
            table.model.layers.resize(static_cast<size_t>(n_layers));
        }
        if (n_embd > 0) table.model.hidden = static_cast<int>(n_embd);
        if (n_vocab > 0) table.model.vocab = static_cast<int>(n_vocab);
        for (int i = 0; i < n_layers; ++i) {
            if (kind[static_cast<size_t>(i)] == 1)
                table.model.layers[static_cast<size_t>(i)].kind = LayerKind::GatedAttn;
            else
                table.model.layers[static_cast<size_t>(i)].kind = LayerKind::GatedDeltaNet;
        }
        if (n_embd > 0) table.model.hidden = static_cast<int>(n_embd);
        if (block_count > 0 && block_count != n_layers) {
            throw LoadError("GGUF block_count mismatch");
        }

        for (const Meta& m : metas) {
            if (is_visual(m.name)) continue;
            std::string ir = map_gguf_name(m.name);
            if (ir.empty()) continue;
            if (opt.max_layers > 0 && ir.rfind("layers[", 0) == 0) {
                const int li = std::stoi(ir.substr(7));
                if (li >= opt.max_layers) continue;
            }
            TensorDesc td;
            td.ir_name = ir;
            td.src_name = m.name;
            td.gguf_type = m.ggml;
            td.quant = ggml_to_quant(m.ggml);
            td.ndim = static_cast<int>(m.shape.size());
            for (int d = 0; d < td.ndim && d < 4; ++d) td.shape[d] = m.shape[static_cast<size_t>(d)];

            r.in.seekg(static_cast<std::streamoff>(data_start + m.offset), std::ios::beg);
            // size from remaining? compute by type
            int64_t ne = 1;
            for (int64_t d : m.shape) ne *= d;
            size_t nbytes = 0;
            switch (m.ggml) {
            case GGML_F32:
                nbytes = static_cast<size_t>(ne) * 4;
                break;
            case GGML_F16:
            case GGML_BF16:
                nbytes = static_cast<size_t>(ne) * 2;
                break;
            case GGML_Q8_0:
                nbytes = static_cast<size_t>((ne / 32) * 34);
                break;
            case GGML_Q4_K:
                nbytes = static_cast<size_t>((ne / 256) * 144);
                break;
            case GGML_Q5_K:
                nbytes = static_cast<size_t>((ne / 256) * 176);
                break;
            case GGML_Q6_K:
                nbytes = static_cast<size_t>((ne / 256) * 210);
                break;
            default:
                nbytes = static_cast<size_t>(ne);
                break;
            }
            td.nbytes = nbytes;
            td.data.resize(nbytes);
            r.in.read(reinterpret_cast<char*>(td.data.data()), static_cast<std::streamsize>(nbytes));
            if (!r.in) throw LoadError("GGUF truncated tensor " + m.name);

            if (opt.reject_quantized_leftover && is_leftover_name(ir) && leftover_quant_forbidden(td.quant)) {
                throw LoadError("leftover over-quantized: " + m.name);
            }
            if (td.quant == QuantKind::IQUnknown && !opt.allow_iq_dequant) {
                throw LoadError("unsupported ggml_type on " + m.name);
            }
            table.tensors.emplace(ir, std::move(td));
        }

        auto need = [&](const std::string& ir) {
            if (!table.find(ir))
                throw LoadError("GGUF missing " + ir + " architecture=" + architecture);
        };
        need("embed");
        if (opt.max_layers > 0 && opt.max_layers < table.model.n_layers) {
            table.model.n_layers = opt.max_layers;
            table.model.layers.resize(static_cast<size_t>(opt.max_layers));
        }
        table.model.has_mtp = table.find("mtp.fc") != nullptr && table.find("mtp.norm") != nullptr;
        for (int i = 0; i < table.model.n_layers; ++i) {
            if (table.model.layers[static_cast<size_t>(i)].kind == LayerKind::GatedDeltaNet) {
                if (!table.find("layers[" + std::to_string(i) + "].delta.leftover.a_log") &&
                    !table.find("layers[" + std::to_string(i) + "].delta.leftover.conv1d")) {
                    throw LoadError("GGUF missing DeltaNet leftovers on blk." + std::to_string(i) +
                                    " architecture=" + architecture);
                }
            }
        }
        return table;
    }
};

} // namespace

std::unique_ptr<IWeightLoader> make_gguf_loader() { return std::make_unique<GgufLoaderImpl>(); }

} // namespace rapidllm
