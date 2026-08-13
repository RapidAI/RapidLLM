#include "frontend/safetensors.h"

#include "frontend/json_mini.h"

#include <fstream>

namespace rapidllm {

static uint64_t read_u64(std::istream& in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 8);
    if (!in) throw LoadError("safetensors: truncated header size");
    return v;
}

SafetensorsFile read_safetensors(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw LoadError("cannot open " + path.string());
    const uint64_t hlen = read_u64(in);
    if (hlen > 64ull * 1024 * 1024) throw LoadError("safetensors header too large");
    std::string header(static_cast<size_t>(hlen), '\0');
    in.read(header.data(), static_cast<std::streamsize>(hlen));
    if (!in) throw LoadError("safetensors: truncated header");
    Json j = parse_json(header);
    if (!j.is_obj()) throw LoadError("safetensors: header not object");

    in.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());
    const uint64_t data_off = 8 + hlen;

    SafetensorsFile out;
    for (const auto& [name, meta] : j.as_obj()) {
        if (name == "__metadata__") continue;
        if (!meta.is_obj()) continue;
        STTensor t;
        t.name = name;
        t.dtype = meta.at("dtype").as_str();
        const JsonArray& sh = meta.at("shape").as_arr();
        for (const Json& d : sh) t.shape.push_back(static_cast<int64_t>(d.as_num()));
        const JsonArray& off = meta.at("data_offsets").as_arr();
        t.begin = static_cast<uint64_t>(off.at(0).as_num());
        t.end = static_cast<uint64_t>(off.at(1).as_num());
        if (data_off + t.end > file_size) throw LoadError("safetensors: data past EOF for " + name);
        t.bytes.resize(static_cast<size_t>(t.end - t.begin));
        in.seekg(static_cast<std::streamoff>(data_off + t.begin), std::ios::beg);
        in.read(reinterpret_cast<char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
        if (!in) throw LoadError("safetensors: truncated tensor " + name);
        out.tensors.emplace(name, std::move(t));
    }
    return out;
}

} // namespace rapidllm
