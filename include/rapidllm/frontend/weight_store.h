#pragma once

#include "rapidllm/ir/model_desc.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rapidllm {

class Device;
struct TensorView;

enum class SourceKind { HfFp8Dir, GgufFile };

struct TensorDesc {
    std::string ir_name;
    std::string src_name;
    QuantKind quant = QuantKind::BF16;
    int64_t shape[4]{};
    int ndim = 0;
    uint64_t nbytes = 0;
    uint64_t file_offset = 0;
    std::string scale_name;
    int gguf_type = -1;
    bool skipped = false;
    std::vector<uint8_t> data;
    std::vector<float> scale;
};

class TensorTable {
public:
    ModelDesc model;
    SourceKind source = SourceKind::HfFp8Dir;
    std::unordered_map<std::string, TensorDesc> tensors;
    const TensorDesc* find(std::string_view ir_name) const;
    TensorDesc* find_mut(std::string_view ir_name);
    uint64_t total_nbytes() const;
};

struct LoadOptions {
    int max_layers = -1;
    bool language_only = true;
    bool mmap = true;
    bool hugepage = false;
    bool repack_int4 = false;
    bool allow_iq_dequant = true;
    bool reject_quantized_leftover = true;
};

class IWeightLoader {
public:
    virtual ~IWeightLoader() = default;
    virtual TensorTable load(const std::filesystem::path& path, const LoadOptions& opt) = 0;
};

std::unique_ptr<IWeightLoader> make_loader(const std::filesystem::path& path);

class WeightStore {
public:
    static WeightStore open(const std::filesystem::path& path, Device& dev, const LoadOptions& opt);
    const ModelDesc& model() const { return table_.model; }
    const TensorTable& table() const { return table_; }
    TensorTable& table() { return table_; }

private:
    TensorTable table_;
};

std::string map_hf_name(std::string_view src);
std::string map_gguf_name(std::string_view src);
bool is_visual(std::string_view name);
bool is_visual_or_mtp(std::string_view name); // visual only (MTP is first-class)
bool is_leftover_name(std::string_view ir_or_src);

class LoadError : public std::runtime_error {
public:
    explicit LoadError(const std::string& m) : std::runtime_error(m) {}
};

} // namespace rapidllm
