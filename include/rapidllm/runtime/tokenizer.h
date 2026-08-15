#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rapidllm {

class Tokenizer {
public:
    explicit Tokenizer(int vocab);
    int encode(std::string_view utf8, int32_t* ids, int cap) const;
    std::string decode(const int32_t* ids, int n) const;
    int vocab() const { return vocab_; }

    static constexpr int kBosEos = 248044;
    static constexpr int kVisionStart = 248053;
    static constexpr int kVisionEnd = 248054;
    static constexpr int kImage = 248056;
    static constexpr int kVideo = 248057;

private:
    int vocab_ = 0;
};

// vision_start + image_pad * n_vis + vision_end + text_ids. Returns n or -1 if cap is short.
int pack_vl_prompt(int n_vis, const int32_t* text, int n_text, int32_t* out, int cap,
                   int vision_start = Tokenizer::kVisionStart, int image_id = Tokenizer::kImage,
                   int vision_end = Tokenizer::kVisionEnd);

std::string apply_chat_template(std::string_view user, bool enable_thinking);

struct ChatTurn {
    std::string role;
    std::string content;
};
std::string apply_chat_messages(const std::vector<ChatTurn>& turns, bool enable_thinking);

} // namespace rapidllm
