#pragma once

#include "rapidllm/ir/model_desc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rapidllm {

enum class DeviceKind { CPU, Vulkan, CUDA };

struct BufferDesc {
    size_t bytes = 0;
    enum class Usage { Weight, Activation, KV, Recurrent, Scratch, HostStaging } usage{};
    bool host_visible = false;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual void* host_ptr() = 0;
    virtual size_t bytes() const = 0;
};

struct TensorView {
    Buffer* buf = nullptr;
    DType dtype = DType::F32;
    int32_t ndim = 0;
    int64_t shape[8]{};
    int64_t stride[8]{};
    uint64_t byte_offset = 0;
    void* host_ptr() const {
        if (!buf) return nullptr;
        return static_cast<uint8_t*>(const_cast<Buffer*>(buf)->host_ptr()) + byte_offset;
    }
};

class Stream {
public:
    virtual ~Stream() = default;
    virtual void begin_step(const char* tag) = 0;
    virtual void end_step() = 0;
    virtual void synchronize() = 0;
};

class Device {
public:
    virtual ~Device() = default;
    virtual DeviceKind kind() const = 0;
    virtual const char* name() const = 0;
    virtual std::unique_ptr<Buffer> allocate(const BufferDesc&) = 0;
    virtual std::unique_ptr<Stream> create_stream() = 0;
    virtual void memcpy(TensorView dst, TensorView src, Stream&) = 0;
    virtual void launch(const char* kernel, const TensorView* args, int nargs,
                        const void* params, size_t params_bytes, Stream&) = 0;
    virtual bool has_kernel(const char* name) const = 0;
};

std::unique_ptr<Device> create_device(DeviceKind k);

class CpuBuffer final : public Buffer {
public:
    explicit CpuBuffer(size_t n);
    void* host_ptr() override { return data_.data(); }
    size_t bytes() const override { return data_.size(); }

private:
    std::vector<uint8_t> data_;
};

} // namespace rapidllm
