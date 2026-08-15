#pragma once

#include <string>
#include <vector>

namespace rapidllm {

struct ImageRgb {
    int h = 0, w = 0;
    std::vector<float> rgb; // H*W*3, RGB in [0, 1]
};

// PPM P6, BMP, PNG, JPEG (stb). Throws std::runtime_error on failure.
ImageRgb load_image_rgb(const std::string& path);

} // namespace rapidllm
