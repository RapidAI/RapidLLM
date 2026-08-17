#include "rapidllm/server/http_serve.h"

#include "frontend/json_mini.h"
#include "rapidllm/runtime/tokenizer.h"
#include "rapidllm/version.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socklen_t = int;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#define closesocket close
#endif

namespace rapidllm::serve {
namespace {

bool append_text_pieces(const Json& node, std::string& out) {
    if (node.is_str()) {
        if (!out.empty()) out.push_back('\n');
        out += node.as_str();
        return true;
    }
    if (node.is_arr()) {
        for (const Json& p : node.as_arr()) {
            if (p.is_str()) {
                if (!out.empty()) out.push_back('\n');
                out += p.as_str();
                continue;
            }
            if (!p.is_obj()) continue;
            const Json* t = p.get("text");
            if (!t && p.get("type") && p.at("type").is_str()) {
                const std::string ty = p.at("type").as_str();
                if (ty == "input_text" || ty == "output_text" || ty == "text") t = p.get("text");
            }
            if (t && t->is_str()) {
                if (!out.empty()) out.push_back('\n');
                out += t->as_str();
            }
        }
        return !out.empty();
    }
    if (node.is_obj()) {
        const Json* t = node.get("text");
        if (t && t->is_str()) {
            if (!out.empty()) out.push_back('\n');
            out += t->as_str();
            return true;
        }
    }
    return false;
}

bool messages_from_array(const Json& arr, std::vector<ChatTurn>& out, std::string& err) {
    if (!arr.is_arr()) {
        err = "messages must be an array";
        return false;
    }
    for (const Json& m : arr.as_arr()) {
        if (!m.is_obj()) continue;
        ChatTurn t;
        if (const Json* r = m.get("role"); r && r->is_str()) t.role = r->as_str();
        else t.role = "user";
        const Json* c = m.get("content");
        if (!c) c = m.get("text");
        if (c) append_text_pieces(*c, t.content);
        if (t.content.empty() && t.role == "system") continue;
        out.push_back(std::move(t));
    }
    if (out.empty()) {
        err = "no messages";
        return false;
    }
    return true;
}

long long now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string make_id(const char* prefix) {
    static int seq = 0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s%lld%x", prefix, static_cast<long long>(now_unix()), ++seq);
    return buf;
}

HttpResponse json_ok(std::string body) {
    HttpResponse r;
    r.status = 200;
    r.body = std::move(body);
    return r;
}

HttpResponse json_err(int status, std::string_view type, std::string_view msg) {
    HttpResponse r;
    r.status = status;
    r.body = render_error(status, type, msg);
    return r;
}

void read_stream_flag(const Json& j, GenRequest& out) {
    const Json* s = j.get("stream");
    if (!s) return;
    if (s->is_bool()) out.stream = s->as_bool();
    else if (s->is_num()) out.stream = s->as_int() != 0;
}

int classify_path(std::string_view path) {
    if (path == "/v1/chat/completions" || path == "/chat/completions") return 1;
    if (path == "/v1/responses" || path == "/responses") return 2;
    if (path == "/v1/messages" || path == "/messages") return 3;
    return 0;
}

std::string decode_ids(RapidLLM* eng, const int32_t* ids, int n) {
    if (!eng || !ids || n <= 0) return {};
    RapidError e{};
    std::string t(static_cast<size_t>(n) * 16 + 32, '\0');
    int dn = rapidllm_decode_ids(eng, ids, n, t.data(), static_cast<int>(t.size()), &e);
    if (dn < 0) {
        t.assign(static_cast<size_t>(n) * 64 + 64, '\0');
        dn = rapidllm_decode_ids(eng, ids, n, t.data(), static_cast<int>(t.size()), &e);
    }
    if (dn < 0) return {};
    t.resize(static_cast<size_t>(dn));
    return t;
}

bool push_sse(std::string& body, SseEmit emit, void* ctx, std::string_view frame) {
    body.append(frame);
    if (emit) return emit(ctx, frame.data(), frame.size());
    return true;
}

bool send_all(int fd, const char* p, size_t n) {
    while (n > 0) {
        const int chunk = n > 1u << 20 ? (1 << 20) : static_cast<int>(n);
        const int w = send(fd, p, chunk, 0);
        if (w <= 0) return false;
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

bool sock_emit(void* ctx, const char* data, size_t n) {
    if (!ctx || !data) return false;
    return send_all(*static_cast<int*>(ctx), data, n);
}

} // namespace

std::string json_escape(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                o += b;
            } else
                o.push_back(static_cast<char>(c));
        }
    }
    return o;
}

bool parse_openai_chat(std::string_view body, GenRequest& out, std::string& err) {
    try {
        const Json j = parse_json(body);
        if (!j.is_obj()) {
            err = "expected object";
            return false;
        }
        if (const Json* m = j.get("model"); m && m->is_str()) out.model = m->as_str();
        if (const Json* t = j.get("max_tokens"); t && t->is_num()) out.max_tokens = t->as_int();
        if (const Json* t = j.get("max_completion_tokens"); t && t->is_num()) out.max_tokens = t->as_int();
        if (const Json* t = j.get("temperature"); t && t->is_num()) out.temperature = static_cast<float>(t->as_num());
        read_stream_flag(j, out);
        const Json* msgs = j.get("messages");
        if (!msgs) {
            err = "missing messages";
            return false;
        }
        return messages_from_array(*msgs, out.messages, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool parse_openai_responses(std::string_view body, GenRequest& out, std::string& err) {
    try {
        const Json j = parse_json(body);
        if (!j.is_obj()) {
            err = "expected object";
            return false;
        }
        if (const Json* m = j.get("model"); m && m->is_str()) out.model = m->as_str();
        if (const Json* t = j.get("max_output_tokens"); t && t->is_num()) out.max_tokens = t->as_int();
        if (const Json* t = j.get("max_tokens"); t && t->is_num()) out.max_tokens = t->as_int();
        if (const Json* t = j.get("temperature"); t && t->is_num()) out.temperature = static_cast<float>(t->as_num());
        read_stream_flag(j, out);
        if (const Json* inp = j.get("input")) {
            if (inp->is_str()) {
                out.messages.push_back({"user", inp->as_str()});
                return true;
            }
            if (inp->is_arr()) return messages_from_array(*inp, out.messages, err);
        }
        if (const Json* msgs = j.get("messages")) return messages_from_array(*msgs, out.messages, err);
        err = "missing input";
        return false;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool parse_anthropic_messages(std::string_view body, GenRequest& out, std::string& err) {
    try {
        const Json j = parse_json(body);
        if (!j.is_obj()) {
            err = "expected object";
            return false;
        }
        if (const Json* m = j.get("model"); m && m->is_str()) out.model = m->as_str();
        if (const Json* t = j.get("max_tokens"); t && t->is_num()) out.max_tokens = t->as_int();
        if (const Json* t = j.get("temperature"); t && t->is_num()) out.temperature = static_cast<float>(t->as_num());
        read_stream_flag(j, out);
        if (const Json* sys = j.get("system")) {
            std::string sys_t;
            append_text_pieces(*sys, sys_t);
            if (!sys_t.empty()) out.messages.push_back({"system", sys_t});
        }
        const Json* msgs = j.get("messages");
        if (!msgs) {
            err = "missing messages";
            return false;
        }
        return messages_from_array(*msgs, out.messages, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

std::string render_openai_chat(const std::string& id, const std::string& model, const std::string& text, int prompt_n,
                               int new_n, long long created) {
    std::ostringstream o;
    o << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",\"created\":" << created
      << ",\"model\":\"" << json_escape(model) << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
      << "\"content\":\"" << json_escape(text) << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
      << prompt_n << ",\"completion_tokens\":" << new_n << ",\"total_tokens\":" << (prompt_n + new_n) << "}}";
    return o.str();
}

std::string render_openai_responses(const std::string& id, const std::string& model, const std::string& text,
                                    int prompt_n, int new_n) {
    std::ostringstream o;
    o << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"response\",\"status\":\"completed\",\"model\":\""
      << json_escape(model) << "\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":["
      << "{\"type\":\"output_text\",\"text\":\"" << json_escape(text) << "\"}]}],\"usage\":{\"input_tokens\":"
      << prompt_n << ",\"output_tokens\":" << new_n << ",\"total_tokens\":" << (prompt_n + new_n) << "}}";
    return o.str();
}

std::string render_anthropic(const std::string& id, const std::string& model, const std::string& text, int prompt_n,
                             int new_n) {
    std::ostringstream o;
    o << "{\"id\":\"" << json_escape(id) << "\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\""
      << json_escape(model) << "\",\"content\":[{\"type\":\"text\",\"text\":\"" << json_escape(text)
      << "\"}],\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":" << prompt_n
      << ",\"output_tokens\":" << new_n << "}}";
    return o.str();
}

std::string render_models(const std::string& model) {
    std::ostringstream o;
    o << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(model)
      << "\",\"object\":\"model\",\"owned_by\":\"rapidllm\"}]}";
    return o.str();
}

std::string render_error(int status, std::string_view type, std::string_view message) {
    std::ostringstream o;
    o << "{\"error\":{\"message\":\"" << json_escape(message) << "\",\"type\":\"" << json_escape(type)
      << "\",\"code\":" << status << "}}";
    return o.str();
}

std::string sse_data(std::string_view payload, std::string_view event) {
    std::string o;
    if (!event.empty()) {
        o += "event: ";
        o += event;
        o += "\n";
    }
    o += "data: ";
    o += payload;
    o += "\n\n";
    return o;
}

HttpResponse handle_http(const HttpRequest& req, RapidLLM* eng, RapidSession* sess, const std::string& model_id) {
    return handle_http(req, eng, sess, model_id, nullptr, nullptr);
}

HttpResponse handle_http(const HttpRequest& req, RapidLLM* eng, RapidSession* sess, const std::string& model_id,
                         SseEmit emit, void* emit_ctx) {
    if (req.method == "GET" && (req.path == "/health" || req.path == "/v1/health")) {
        return json_ok("{\"status\":\"ok\"}");
    }
    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        return json_ok(render_models(model_id));
    }
    if (req.method != "POST") return json_err(405, "invalid_request_error", "method not allowed");

    const bool is_chat = req.path == "/v1/chat/completions" || req.path == "/chat/completions";
    const bool is_resp = req.path == "/v1/responses" || req.path == "/responses";
    const bool is_anth = req.path == "/v1/messages" || req.path == "/messages";
    if (!is_chat && !is_resp && !is_anth)
        return json_err(404, "invalid_request_error", "unknown route");

    GenRequest gr;
    std::string err;
    bool ok = false;
    if (is_chat) ok = parse_openai_chat(req.body, gr, err);
    else if (is_resp) ok = parse_openai_responses(req.body, gr, err);
    else ok = parse_anthropic_messages(req.body, gr, err);
    if (!ok) return json_err(400, "invalid_request_error", err);

    if (gr.max_tokens <= 0) gr.max_tokens = 16;
    if (gr.max_tokens > 4096) gr.max_tokens = 4096;
    const std::string model = gr.model.empty() ? model_id : gr.model;

    std::vector<rapidllm::ChatTurn> turns;
    turns.reserve(gr.messages.size());
    for (const auto& m : gr.messages) turns.push_back({m.role, m.content});
    const std::string prompt = apply_chat_messages(turns, false);

    RapidError e{};
    std::vector<int32_t> ids(static_cast<size_t>(prompt.size()) + 8);
    const int n = rapidllm_encode(eng, prompt.c_str(), ids.data(), static_cast<int>(ids.size()), &e);
    if (n <= 0) return json_err(400, "invalid_request_error", "encode failed");

    rapidllm_session_set_max_new(sess, gr.max_tokens);
    RapidSampleParams sp{};
    sp.greedy = gr.temperature <= 0.f ? 1 : 0;
    sp.temperature = gr.temperature;

    if (!gr.stream) {
        std::vector<int32_t> out(static_cast<size_t>(gr.max_tokens));
        const int got = rapidllm_generate(sess, ids.data(), n, &sp, out.data(), gr.max_tokens, &e);
        if (got < 0) return json_err(500, "server_error", e.message);
        const std::string text = decode_ids(eng, out.data(), got);
        if (is_chat)
            return json_ok(render_openai_chat(make_id("chatcmpl-"), model, text, n, got, now_unix()));
        if (is_resp) return json_ok(render_openai_responses(make_id("resp_"), model, text, n, got));
        return json_ok(render_anthropic(make_id("msg_"), model, text, n, got));
    }

    if (rapidllm_prefill(sess, ids.data(), n, &e) != RAPID_OK)
        return json_err(500, "server_error", e.message[0] ? e.message : "prefill failed");

    const long long created = now_unix();
    const std::string id = is_chat ? make_id("chatcmpl-") : (is_resp ? make_id("resp_") : make_id("msg_"));
    HttpResponse r;
    r.status = 200;
    r.content_type = "text/event-stream";
    auto fail = [&](std::string_view msg) {
        const std::string ev = sse_data(render_error(500, "server_error", msg));
        push_sse(r.body, emit, emit_ctx, ev);
        return r;
    };

    if (is_chat) {
        std::ostringstream o;
        o << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
          << ",\"model\":\"" << json_escape(model) << "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
          << "\"finish_reason\":null}]}";
        if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str()))) return r;
    } else if (is_resp) {
        std::ostringstream o;
        o << "{\"type\":\"response.created\",\"response\":{\"id\":\"" << json_escape(id)
          << "\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"" << json_escape(model) << "\"}}";
        if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str(), "response.created"))) return r;
    } else {
        std::ostringstream o;
        o << "{\"type\":\"message_start\",\"message\":{\"id\":\"" << json_escape(id)
          << "\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" << json_escape(model)
          << "\",\"content\":[],\"stop_reason\":null}}";
        if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str(), "message_start"))) return r;
        if (!push_sse(r.body, emit, emit_ctx,
                      sse_data("{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\","
                               "\"text\":\"\"}}",
                               "content_block_start")))
            return r;
    }

    std::vector<int32_t> toks;
    toks.reserve(static_cast<size_t>(gr.max_tokens));
    std::string prev;
    for (int i = 0; i < gr.max_tokens; ++i) {
        int32_t tok = 0;
        if (rapidllm_sample(sess, &sp, &tok, &e) != RAPID_OK) return fail(e.message);
        toks.push_back(tok);
        const std::string cur = decode_ids(eng, toks.data(), static_cast<int>(toks.size()));
        std::string delta;
        if (cur.size() >= prev.size() && cur.compare(0, prev.size(), prev) == 0)
            delta = cur.substr(prev.size());
        else
            delta = cur;
        prev = cur;
        if (!delta.empty()) {
            if (is_chat) {
                std::ostringstream o;
                o << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",\"created\":"
                  << created << ",\"model\":\"" << json_escape(model)
                  << "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" << json_escape(delta)
                  << "\"},\"finish_reason\":null}]}";
                if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str()))) return r;
            } else if (is_resp) {
                std::ostringstream o;
                o << "{\"type\":\"response.output_text.delta\",\"delta\":\"" << json_escape(delta) << "\"}";
                if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str(), "response.output_text.delta"))) return r;
            } else {
                std::ostringstream o;
                o << "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\""
                  << json_escape(delta) << "\"}}";
                if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str(), "content_block_delta"))) return r;
            }
        }
        if (i + 1 < gr.max_tokens) {
            if (rapidllm_decode(sess, tok, nullptr, &e) != RAPID_OK) return fail(e.message);
        }
    }
    const int got = static_cast<int>(toks.size());

    if (is_chat) {
        std::ostringstream o;
        o << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",\"created\":" << created
          << ",\"model\":\"" << json_escape(model)
          << "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
          << n << ",\"completion_tokens\":" << got << ",\"total_tokens\":" << (n + got) << "}}";
        if (!push_sse(r.body, emit, emit_ctx, sse_data(o.str()))) return r;
        push_sse(r.body, emit, emit_ctx, sse_data("[DONE]"));
    } else if (is_resp) {
        std::ostringstream o;
        o << "{\"type\":\"response.completed\",\"response\":{\"id\":\"" << json_escape(id)
          << "\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"" << json_escape(model)
          << "\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\","
          << "\"text\":\"" << json_escape(prev) << "\"}]}],\"usage\":{\"input_tokens\":" << n
          << ",\"output_tokens\":" << got << ",\"total_tokens\":" << (n + got) << "}}}";
        push_sse(r.body, emit, emit_ctx, sse_data(o.str(), "response.completed"));
    } else {
        if (!push_sse(r.body, emit, emit_ctx, sse_data("{\"type\":\"content_block_stop\",\"index\":0}",
                                                      "content_block_stop")))
            return r;
        std::ostringstream d;
        d << "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":"
          << got << "}}";
        if (!push_sse(r.body, emit, emit_ctx, sse_data(d.str(), "message_delta"))) return r;
        push_sse(r.body, emit, emit_ctx, sse_data("{\"type\":\"message_stop\"}", "message_stop"));
    }
    return r;
}

#if defined(_WIN32)
struct WinsockOnce {
    WinsockOnce() {
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
    }
    ~WinsockOnce() { WSACleanup(); }
};
static WinsockOnce g_wsa;
#endif

int serve_listen(const char* host, int port, RapidLLM* eng, RapidSession* sess, const std::string& model_id) {
    const char* bind_host = host && host[0] ? host : "127.0.0.1";
    int fd = static_cast<int>(socket(AF_INET, SOCK_STREAM, 0));
    if (fd < 0) {
        std::fprintf(stderr, "socket failed\n");
        return 1;
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, bind_host, &addr.sin_addr) != 1) {
        std::fprintf(stderr, "bad host %s\n", bind_host);
        closesocket(fd);
        return 1;
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "bind %s:%d failed\n", bind_host, port);
        closesocket(fd);
        return 1;
    }
    if (listen(fd, 16) != 0) {
        std::fprintf(stderr, "listen failed\n");
        closesocket(fd);
        return 1;
    }
    std::fprintf(stderr,
                 "rapidllm serve %s:%d  /v1/chat/completions  /v1/responses  /v1/messages  (SSE stream=true)\n",
                 bind_host, port);
    std::fflush(stderr);
    for (;;) {
        sockaddr_in cli{};
        socklen_t cl = sizeof(cli);
        int cfd = static_cast<int>(accept(fd, reinterpret_cast<sockaddr*>(&cli), &cl));
        if (cfd < 0) continue;
        std::string raw;
        char buf[4096];
        for (;;) {
            const int n = recv(cfd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            raw.append(buf, static_cast<size_t>(n));
            if (raw.find("\r\n\r\n") != std::string::npos) break;
            if (raw.size() > 1u << 20) break;
        }
        HttpRequest hr;
        const size_t hdr_end = raw.find("\r\n\r\n");
        std::string headers = hdr_end == std::string::npos ? raw : raw.substr(0, hdr_end);
        std::string body = hdr_end == std::string::npos ? std::string() : raw.substr(hdr_end + 4);
        {
            std::istringstream ls(headers);
            std::string line;
            if (std::getline(ls, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::istringstream fl(line);
                fl >> hr.method >> hr.path;
            }
            int content_len = 0;
            while (std::getline(ls, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                const auto c = line.find(':');
                if (c == std::string::npos) continue;
                std::string k = line.substr(0, c);
                for (char& ch : k)
                    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
                if (k == "content-length") content_len = std::atoi(line.c_str() + c + 1);
            }
            while (static_cast<int>(body.size()) < content_len) {
                const int n = recv(cfd, buf, sizeof(buf), 0);
                if (n <= 0) break;
                body.append(buf, static_cast<size_t>(n));
            }
            hr.body = std::move(body);
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));

        bool live_sse = false;
        if (hr.method == "POST") {
            const int kind = classify_path(hr.path);
            if (kind) {
                GenRequest peek;
                std::string perr;
                bool pok = false;
                if (kind == 1) pok = parse_openai_chat(hr.body, peek, perr);
                else if (kind == 2) pok = parse_openai_responses(hr.body, peek, perr);
                else pok = parse_anthropic_messages(hr.body, peek, perr);
                live_sse = pok && peek.stream;
            }
        }

        if (live_sse) {
            const char* hdr = "HTTP/1.1 200 OK\r\n"
                              "Content-Type: text/event-stream\r\n"
                              "Cache-Control: no-cache\r\n"
                              "Connection: close\r\n"
                              "X-Accel-Buffering: no\r\n\r\n";
            send_all(cfd, hdr, std::strlen(hdr));
            const HttpResponse resp = handle_http(hr, eng, sess, model_id, sock_emit, &cfd);
            if (resp.status != 200 && !resp.body.empty()) {
                const std::string ev = sse_data(resp.body);
                send_all(cfd, ev.data(), ev.size());
            }
            closesocket(cfd);
            continue;
        }

        const HttpResponse resp = handle_http(hr, eng, sess, model_id);
        std::ostringstream out;
        out << "HTTP/1.1 " << resp.status << (resp.status == 200 ? " OK" : " ERR") << "\r\n"
            << "Content-Type: " << resp.content_type << "\r\n"
            << "Content-Length: " << resp.body.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << resp.body;
        const std::string wire = out.str();
        send_all(cfd, wire.data(), wire.size());
        closesocket(cfd);
    }
}

} // namespace rapidllm::serve
