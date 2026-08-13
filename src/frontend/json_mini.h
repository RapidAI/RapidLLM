#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rapidllm {

struct Json;

using JsonArray = std::vector<Json>;
using JsonObject = std::map<std::string, Json>;

struct Json {
    std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject> v;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_num() const { return std::holds_alternative<double>(v); }
    bool is_str() const { return std::holds_alternative<std::string>(v); }
    bool is_arr() const { return std::holds_alternative<JsonArray>(v); }
    bool is_obj() const { return std::holds_alternative<JsonObject>(v); }

    bool as_bool() const { return std::get<bool>(v); }
    double as_num() const { return std::get<double>(v); }
    int as_int() const { return static_cast<int>(std::get<double>(v)); }
    const std::string& as_str() const { return std::get<std::string>(v); }
    const JsonArray& as_arr() const { return std::get<JsonArray>(v); }
    const JsonObject& as_obj() const { return std::get<JsonObject>(v); }

    const Json* get(const char* key) const {
        if (!is_obj()) return nullptr;
        auto it = as_obj().find(key);
        return it == as_obj().end() ? nullptr : &it->second;
    }
    const Json& at(const char* key) const {
        const Json* p = get(key);
        if (!p) throw std::runtime_error(std::string("json missing key: ") + key);
        return *p;
    }
};

Json parse_json(std::string_view text);

} // namespace rapidllm
