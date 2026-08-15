#include "rapidllm/runtime/image_io.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace rapidllm {
namespace {

ImageRgb from_u8(const unsigned char* p, int w, int h, int c) {
    ImageRgb im;
    im.w = w;
    im.h = h;
    im.rgb.resize(static_cast<size_t>(h) * w * 3);
    for (int i = 0; i < h * w; ++i) {
        if (c >= 3) {
            im.rgb[static_cast<size_t>(i) * 3 + 0] = p[static_cast<size_t>(i) * c + 0] / 255.f;
            im.rgb[static_cast<size_t>(i) * 3 + 1] = p[static_cast<size_t>(i) * c + 1] / 255.f;
            im.rgb[static_cast<size_t>(i) * 3 + 2] = p[static_cast<size_t>(i) * c + 2] / 255.f;
        } else {
            const float g = p[static_cast<size_t>(i) * c] / 255.f;
            im.rgb[static_cast<size_t>(i) * 3 + 0] = g;
            im.rgb[static_cast<size_t>(i) * 3 + 1] = g;
            im.rgb[static_cast<size_t>(i) * 3 + 2] = g;
        }
    }
    return im;
}

ImageRgb load_ppm(const std::string& path) {
    FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, path.c_str(), "rb") != 0) f = nullptr;
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) throw std::runtime_error("cannot open image " + path);
    char magic[3] = {};
    if (std::fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '6') {
        std::fclose(f);
        throw std::runtime_error("not a P6 PPM: " + path);
    }
    auto skip = [&]() {
        int ch = std::fgetc(f);
        while (ch == '#') {
            while (ch != EOF && ch != '\n') ch = std::fgetc(f);
            ch = std::fgetc(f);
        }
        while (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') ch = std::fgetc(f);
        if (ch != EOF) std::ungetc(ch, f);
    };
    skip();
    int w = 0, h = 0, maxv = 0;
    if (std::fscanf(f, "%d", &w) != 1) {
        std::fclose(f);
        throw std::runtime_error("bad PPM header");
    }
    skip();
    if (std::fscanf(f, "%d", &h) != 1) {
        std::fclose(f);
        throw std::runtime_error("bad PPM header");
    }
    skip();
    if (std::fscanf(f, "%d", &maxv) != 1 || maxv <= 0) {
        std::fclose(f);
        throw std::runtime_error("bad PPM header");
    }
    std::fgetc(f);
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) {
        std::fclose(f);
        throw std::runtime_error("PPM size out of range");
    }
    std::vector<unsigned char> raw(static_cast<size_t>(w) * h * 3);
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) {
        std::fclose(f);
        throw std::runtime_error("PPM truncated");
    }
    std::fclose(f);
    const float s = 1.f / static_cast<float>(maxv);
    ImageRgb im;
    im.w = w;
    im.h = h;
    im.rgb.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) im.rgb[i] = raw[i] * s;
    return im;
}

} // namespace

ImageRgb load_image_rgb(const std::string& path) {
    if (path.size() >= 4) {
        const std::string ext = path.substr(path.size() - 4);
        if (ext == ".ppm" || ext == ".PPM") return load_ppm(path);
    }
    int w = 0, h = 0, c = 0;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!p) {
        const char* why = stbi_failure_reason();
        throw std::runtime_error(std::string("image decode failed: ") + path + " (" +
                                 (why ? why : "unknown") + ")");
    }
    ImageRgb im = from_u8(p, w, h, 3);
    stbi_image_free(p);
    return im;
}

} // namespace rapidllm
