#include "rapidllm/runtime/dual_cache.h"

#include <cstring>

namespace rapidllm {

DualCache::DualCache(Device& dev, const ModelDesc& m, int max_pos, DType) : model_(&m), max_pos_(max_pos) {
    for (const LayerDesc& L : m.layers) {
        if (L.kind == LayerKind::GatedAttn) {
            ++n_attn_;
            kv_heads_ = L.attn.n_kv;
            head_dim_ = L.attn.head_dim;
        } else if (L.kind == LayerKind::GatedDeltaNet) {
            ++n_delta_;
            nv_ = L.delta.n_v_heads;
            dk_ = L.delta.k_dim;
            dv_ = L.delta.v_dim;
            conv_k_ = L.delta.conv_k;
            conv_dim_ = L.delta.n_k_heads * L.delta.k_dim * 2 + L.delta.n_v_heads * L.delta.v_dim;
        }
    }
    kv_stride_ = kv_heads_ * head_dim_;
    BufferDesc kv;
    kv.bytes = static_cast<size_t>(n_attn_) * max_pos_ * kv_stride_ * sizeof(float) * 2;
    kv.usage = BufferDesc::Usage::KV;
    kv.host_visible = true;
    kv_ = dev.allocate(kv);

    BufferDesc st;
    st.bytes = static_cast<size_t>(n_delta_) * nv_ * dk_ * dv_ * sizeof(float);
    st.usage = BufferDesc::Usage::Recurrent;
    st.host_visible = true;
    state_ = dev.allocate(st);

    BufferDesc cv;
    cv.bytes = static_cast<size_t>(n_delta_) * conv_dim_ * conv_k_ * sizeof(float);
    cv.usage = BufferDesc::Usage::Recurrent;
    cv.host_visible = true;
    conv_ = dev.allocate(cv);
    zero();
}

void DualCache::zero() {
    if (kv_) std::memset(kv_->host_ptr(), 0, kv_->bytes());
    if (state_) std::memset(state_->host_ptr(), 0, state_->bytes());
    if (conv_) std::memset(conv_->host_ptr(), 0, conv_->bytes());
}

float* DualCache::k_ptr(int full_attn_index) {
    auto* base = static_cast<float*>(kv_->host_ptr());
    const size_t layer = static_cast<size_t>(full_attn_index) * max_pos_ * kv_stride_;
    return base + layer;
}

float* DualCache::v_ptr(int full_attn_index) {
    auto* base = static_cast<float*>(kv_->host_ptr());
    const size_t half = static_cast<size_t>(n_attn_) * max_pos_ * kv_stride_;
    const size_t layer = static_cast<size_t>(full_attn_index) * max_pos_ * kv_stride_;
    return base + half + layer;
}

float* DualCache::state_ptr(int delta_index) {
    auto* base = static_cast<float*>(state_->host_ptr());
    return base + static_cast<size_t>(delta_index) * nv_ * dk_ * dv_;
}

float* DualCache::conv_ptr(int delta_index) {
    auto* base = static_cast<float*>(conv_->host_ptr());
    return base + static_cast<size_t>(delta_index) * conv_dim_ * conv_k_;
}

int DualCache::conv_dim(int) const { return conv_dim_; }

void DualCache::append_kv(int full_attn_index, const float* k_t, const float* v_t, int pos) {
    std::memcpy(k_ptr(full_attn_index) + static_cast<size_t>(pos) * kv_stride_, k_t,
                static_cast<size_t>(kv_stride_) * sizeof(float));
    std::memcpy(v_ptr(full_attn_index) + static_cast<size_t>(pos) * kv_stride_, v_t,
                static_cast<size_t>(kv_stride_) * sizeof(float));
}

} // namespace rapidllm
