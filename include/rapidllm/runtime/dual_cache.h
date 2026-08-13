#pragma once

#include "rapidllm/backend/device.h"
#include "rapidllm/ir/model_desc.h"

#include <memory>
#include <vector>

namespace rapidllm {

class DualCache {
public:
    DualCache(Device& dev, const ModelDesc& m, int max_pos, DType kv_dtype);

    float* k_ptr(int full_attn_index);
    float* v_ptr(int full_attn_index);
    float* state_ptr(int delta_index);
    float* conv_ptr(int delta_index);

    int max_pos() const { return max_pos_; }
    int kv_stride() const { return kv_stride_; }
    int conv_dim(int delta_index) const;

    void append_kv(int full_attn_index, const float* k_t, const float* v_t, int pos);
    void zero();

private:
    const ModelDesc* model_ = nullptr;
    int max_pos_ = 0;
    int n_attn_ = 0;
    int n_delta_ = 0;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    int kv_stride_ = 0;
    int nv_ = 0, dk_ = 0, dv_ = 0, conv_k_ = 0, conv_dim_ = 0;
    std::unique_ptr<Buffer> kv_;
    std::unique_ptr<Buffer> state_;
    std::unique_ptr<Buffer> conv_;
};

} // namespace rapidllm
