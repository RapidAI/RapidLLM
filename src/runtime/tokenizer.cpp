#include "rapidllm/runtime/tokenizer.h"

#include <sstream>

namespace rapidllm {

Tokenizer::Tokenizer(int vocab) : vocab_(vocab) {}

int Tokenizer::encode(std::string_view utf8, int32_t* ids, int cap) const {
    int n = 0;
    // Tiny / fixture path: if the prompt is comma-separated integers, parse them.
    bool numeric = !utf8.empty();
    for (char c : utf8) {
        if (!(c == ',' || c == ' ' || (c >= '0' && c <= '9'))) {
            numeric = false;
            break;
        }
    }
    if (numeric) {
        std::string tmp(utf8);
        std::stringstream ss(tmp);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            const int id = std::stoi(tok);
            if (n < cap) ids[n] = id % vocab_;
            ++n;
        }
        return n;
    }
    for (unsigned char c : utf8) {
        if (n < cap) ids[n] = static_cast<int32_t>(c % vocab_);
        ++n;
    }
    return n;
}

std::string Tokenizer::decode(const int32_t* ids, int n) const {
    std::string o;
    o.reserve(static_cast<size_t>(n) * 4);
    for (int i = 0; i < n; ++i) {
        if (i) o.push_back(' ');
        o += std::to_string(ids[i]);
    }
    return o;
}

std::string apply_chat_template(std::string_view user, bool enable_thinking) {
    std::string s = "<|im_start|>user\n";
    s.append(user);
    s += "<|im_end|>\n<|im_start|>assistant\n";
    if (enable_thinking) s += "<think>\n";
    return s;
}

} // namespace rapidllm
