#include "rapidllm/backend/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace rapidllm {

CpuBuffer::CpuBuffer(size_t n) : data_(n, 0) {}

namespace {

class CpuStream final : public Stream {
public:
    void begin_step(const char*) override {}
    void end_step() override {}
    void synchronize() override {}
};

class CpuDevice final : public Device {
public:
    DeviceKind kind() const override { return DeviceKind::CPU; }
    const char* name() const override { return "cpu"; }
    std::unique_ptr<Buffer> allocate(const BufferDesc& d) override {
        return std::make_unique<CpuBuffer>(d.bytes);
    }
    std::unique_ptr<Stream> create_stream() override { return std::make_unique<CpuStream>(); }
    void memcpy(TensorView dst, TensorView src, Stream&) override {
        const size_t n = src.buf && dst.buf ? std::min(src.buf->bytes(), dst.buf->bytes()) : 0;
        if (n && dst.host_ptr() && src.host_ptr()) std::memcpy(dst.host_ptr(), src.host_ptr(), n);
    }
    void launch(const char*, const TensorView*, int, const void*, size_t, Stream&) override {}
    bool has_kernel(const char*) const override { return true; }
};

} // namespace

std::unique_ptr<Device> create_device(DeviceKind k) {
    if (k != DeviceKind::CPU) throw std::runtime_error("only CPU device is implemented in this increment");
    return std::make_unique<CpuDevice>();
}

} // namespace rapidllm
