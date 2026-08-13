#include "frontend/json_mini.h"

#include <cctype>

namespace rapidllm {
namespace {

struct Parser {
    std::string_view s;
    size_t i = 0;

    [[noreturn]] void err(const char* m) const { throw std::runtime_error(std::string("json: ") + m); }

    void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }

    char peek() {
        skip();
        if (i >= s.size()) err("unexpected end");
        return s[i];
    }

    char getc() {
        skip();
        if (i >= s.size()) err("unexpected end");
        return s[i++];
    }

    Json parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Json{parse_string()};
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    Json parse_object() {
        if (getc() != '{') err("expected {");
        JsonObject o;
        skip();
        if (peek() == '}') {
            ++i;
            return Json{std::move(o)};
        }
        for (;;) {
            if (peek() != '"') err("expected string key");
            std::string key = parse_string();
            if (getc() != ':') err("expected :");
            o.emplace(std::move(key), parse_value());
            char c = getc();
            if (c == '}') break;
            if (c != ',') err("expected , or }");
        }
        return Json{std::move(o)};
    }

    Json parse_array() {
        if (getc() != '[') err("expected [");
        JsonArray a;
        skip();
        if (peek() == ']') {
            ++i;
            return Json{std::move(a)};
        }
        for (;;) {
            a.push_back(parse_value());
            char c = getc();
            if (c == ']') break;
            if (c != ',') err("expected , or ]");
        }
        return Json{std::move(a)};
    }

    std::string parse_string() {
        if (getc() != '"') err("expected \"");
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return out;
            if (c == '\\') {
                if (i >= s.size()) err("bad escape");
                char e = s[i++];
                switch (e) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(e);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    if (i + 4 > s.size()) err("bad unicode");
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[i++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code += h - '0';
                        else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
                        else err("bad hex");
                    }
                    if (code < 0x80) out.push_back(static_cast<char>(code));
                    else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default:
                    err("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        err("unterminated string");
    }

    Json parse_number() {
        skip();
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        if (i < s.size() && s[i] == '.') {
            ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        return Json{std::stod(std::string(s.substr(start, i - start)))};
    }

    Json parse_bool() {
        if (s.substr(i, 4) == "true") {
            i += 4;
            return Json{true};
        }
        if (s.substr(i, 5) == "false") {
            i += 5;
            return Json{false};
        }
        err("bad bool");
    }

    Json parse_null() {
        if (s.substr(i, 4) == "null") {
            i += 4;
            return Json{nullptr};
        }
        err("bad null");
    }
};

} // namespace

Json parse_json(std::string_view text) {
    Parser p{text, 0};
    Json j = p.parse_value();
    p.skip();
    return j;
}

} // namespace rapidllm
