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

int pack_vl_prompt(int n_vis, const int32_t* text, int n_text, int32_t* out, int cap, int vision_start,
                   int image_id, int vision_end) {
    if (n_vis < 0 || n_text < 0 || !out) return -1;
    if (n_text > 0 && !text) return -1;
    const int need = 2 + n_vis + n_text;
    if (need > cap) return -1;
    int i = 0;
    out[i++] = vision_start;
    for (int k = 0; k < n_vis; ++k) out[i++] = image_id;
    out[i++] = vision_end;
    for (int k = 0; k < n_text; ++k) out[i++] = text[k];
    return i;
}

std::string apply_chat_template(std::string_view user, bool enable_thinking) {
    std::string s = "<|im_start|>user\n";
    s.append(user);
    s += "<|im_end|>\n<|im_start|>assistant\n";
    if (enable_thinking) s += "<think>\n";
    return s;
}

std::string apply_chat_messages(const std::vector<ChatTurn>& turns, bool enable_thinking) {
    std::string s;
    bool has_assistant_tail = false;
    for (const auto& t : turns) {
        std::string role = t.role;
        if (role == "human") role = "user";
        if (role == "ai") role = "assistant";
        if (role.empty()) role = "user";
        s += "<|im_start|>";
        s += role;
        s += "\n";
        s += t.content;
        s += "<|im_end|>\n";
        has_assistant_tail = role == "assistant";
    }
    if (!has_assistant_tail) s += "<|im_start|>assistant\n";
    if (enable_thinking) s += "<think>\n";
    return s;
}

} // namespace rapidllm
