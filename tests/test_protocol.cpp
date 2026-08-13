#include "rapidllm/api.h"
#include "rapidllm/server/http_serve.h"

#include <cstdio>
#include <cstring>
#include <string>

#ifndef RAPIDLLM_FIXTURE_DIR
#define RAPIDLLM_FIXTURE_DIR "."
#endif

static int fails = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "fail: %s\n", msg);
        ++fails;
    }
}

int main() {
    using namespace rapidllm::serve;
    std::string err;
    {
        GenRequest r;
        const char* body =
            "{\"model\":\"qwen\",\"max_tokens\":16,\"messages\":["
            "{\"role\":\"system\",\"content\":\"be brief\"},"
            "{\"role\":\"user\",\"content\":\"hello\"}]}";
        expect(parse_openai_chat(body, r, err), "parse openai chat");
        expect(r.messages.size() == 2, "chat two turns");
        expect(r.messages[0].role == "system" && r.messages[1].content == "hello", "chat fields");
        expect(r.max_tokens == 16, "chat max_tokens");
    }
    {
        GenRequest r;
        const char* body = "{\"model\":\"qwen\",\"max_output_tokens\":8,\"input\":\"ping\"}";
        expect(parse_openai_responses(body, r, err), "parse responses string");
        expect(r.messages.size() == 1 && r.messages[0].content == "ping", "responses input str");
        expect(r.max_tokens == 8, "responses max_output_tokens");
    }
    {
        GenRequest r;
        const char* body =
            "{\"model\":\"qwen\",\"input\":[{\"role\":\"user\",\"content\":["
            "{\"type\":\"input_text\",\"text\":\"hi there\"}]}]}";
        expect(parse_openai_responses(body, r, err), "parse responses parts");
        expect(!r.messages.empty() && r.messages[0].content.find("hi there") != std::string::npos,
               "responses input_text");
    }
    {
        GenRequest r;
        const char* body =
            "{\"model\":\"claude\",\"max_tokens\":32,\"system\":\"sys\","
            "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"yo\"}]}]}";
        expect(parse_anthropic_messages(body, r, err), "parse anthropic");
        expect(r.messages.size() == 2, "anthropic system+user");
        expect(r.messages[0].role == "system" && r.messages[1].content == "yo", "anthropic fields");
    }

    const std::string chat = render_openai_chat("chatcmpl-1", "qwen", "ok", 3, 2, 1);
    expect(chat.find("\"object\":\"chat.completion\"") != std::string::npos, "render chat object");
    expect(chat.find("\"content\":\"ok\"") != std::string::npos, "render chat content");
    const std::string resp = render_openai_responses("resp_1", "qwen", "ok", 3, 2);
    expect(resp.find("\"object\":\"response\"") != std::string::npos, "render responses object");
    expect(resp.find("output_text") != std::string::npos, "render responses output_text");
    const std::string anth = render_anthropic("msg_1", "qwen", "ok", 3, 2);
    expect(anth.find("\"type\":\"message\"") != std::string::npos, "render anthropic type");
    expect(anth.find("end_turn") != std::string::npos, "render anthropic stop");

    HttpResponse h = handle_http({"GET", "/health", ""}, nullptr, nullptr, "qwen");
    expect(h.status == 200 && h.body.find("ok") != std::string::npos, "GET /health");
    HttpResponse m = handle_http({"GET", "/v1/models", ""}, nullptr, nullptr, "qwen-local");
    expect(m.status == 200 && m.body.find("qwen-local") != std::string::npos, "GET /v1/models");

    RapidConfig cfg{};
    const std::string fx = std::string(RAPIDLLM_FIXTURE_DIR) + "/tiny_hybrid";
    cfg.model_path = fx.c_str();
    cfg.device = "cpu";
    cfg.ctx = 64;
    cfg.fuse = 1;
    cfg.language_only = 1;
    RapidError e{};
    RapidLLM* eng = rapidllm_load(&cfg, &e);
    expect(eng != nullptr, "load tiny_hybrid");
    if (eng) {
        RapidSessionConfig sc{};
        sc.enable_thinking = 0;
        sc.max_new_tokens = 6;
        sc.spec = 0;
        RapidSession* sess = rapidllm_session_new(eng, &sc, &e);
        expect(sess != nullptr, "session");
        if (sess) {
            HttpRequest rq;
            rq.method = "POST";
            rq.path = "/v1/chat/completions";
            rq.body = "{\"model\":\"tiny\",\"max_tokens\":6,\"messages\":[{\"role\":\"user\",\"content\":\"1,2,3\"}]}";
            HttpResponse out = handle_http(rq, eng, sess, "tiny");
            expect(out.status == 200, "chat generate status");
            expect(out.body.find("chat.completion") != std::string::npos, "chat generate object");
            expect(out.body.find("assistant") != std::string::npos, "chat generate role");

            rq.path = "/v1/responses";
            rq.body = "{\"model\":\"tiny\",\"max_output_tokens\":6,\"input\":\"1,2,3\"}";
            out = handle_http(rq, eng, sess, "tiny");
            expect(out.status == 200 && out.body.find("\"object\":\"response\"") != std::string::npos,
                   "responses generate");

            rq.path = "/v1/messages";
            rq.body = "{\"model\":\"tiny\",\"max_tokens\":6,\"messages\":[{\"role\":\"user\",\"content\":\"1,2,3\"}]}";
            out = handle_http(rq, eng, sess, "tiny");
            expect(out.status == 200 && out.body.find("\"type\":\"message\"") != std::string::npos,
                   "anthropic generate");

            rapidllm_session_free(sess);
        }
        rapidllm_free(eng);
    }

    if (fails) {
        std::fprintf(stderr, "test_protocol %d failure(s)\n", fails);
        return 1;
    }
    std::printf("test_protocol ok openai_chat openai_responses anthropic messages\n");
    return 0;
}
