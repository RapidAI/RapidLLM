#pragma once

#include "rapidllm/api.h"

#include <string>
#include <string_view>
#include <vector>

namespace rapidllm::serve {

struct ChatTurn {
    std::string role;
    std::string content;
};

struct GenRequest {
    std::string model;
    std::vector<ChatTurn> messages;
    int max_tokens = 16;
    float temperature = 0.f;
    bool stream = false;
};

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

bool parse_openai_chat(std::string_view body, GenRequest& out, std::string& err);
bool parse_openai_responses(std::string_view body, GenRequest& out, std::string& err);
bool parse_anthropic_messages(std::string_view body, GenRequest& out, std::string& err);

std::string json_escape(std::string_view s);
std::string render_openai_chat(const std::string& id, const std::string& model, const std::string& text,
                               int prompt_n, int new_n, long long created);
std::string render_openai_responses(const std::string& id, const std::string& model, const std::string& text,
                                    int prompt_n, int new_n);
std::string render_anthropic(const std::string& id, const std::string& model, const std::string& text, int prompt_n,
                             int new_n);
std::string render_models(const std::string& model);
std::string render_error(int status, std::string_view type, std::string_view message);

HttpResponse handle_http(const HttpRequest& req, RapidLLM* eng, RapidSession* sess, const std::string& model_id);

int serve_listen(const char* host, int port, RapidLLM* eng, RapidSession* sess, const std::string& model_id);

} // namespace rapidllm::serve
