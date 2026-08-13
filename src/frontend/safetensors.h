#pragma once

#include "rapidllm/frontend/weight_store.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace rapidllm {

struct STTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
    std::vector<uint8_t> bytes;
};

struct SafetensorsFile {
    std::unordered_map<std::string, STTensor> tensors;
};

SafetensorsFile read_safetensors(const std::filesystem::path& path);

} // namespace rapidllm
