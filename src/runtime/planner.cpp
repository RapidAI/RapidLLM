#include "rapidllm/runtime/planner.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace rapidllm {

uint64_t detect_available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return static_cast<uint64_t>(s.ullAvailPhys);
    return 8ull << 30;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psz > 0) return static_cast<uint64_t>(pages) * static_cast<uint64_t>(psz);
    return 8ull << 30;
#endif
}

PlannerResult plan_memory(const PlannerInput& in) {
    PlannerResult r;
    const int kv_bytes_per_tok = in.n_attn_layers * in.n_kv_heads * in.head_dim * 2 * (in.kv_i8 ? 1 : 2);
    r.kv_bytes = static_cast<uint64_t>(kv_bytes_per_tok) * static_cast<uint64_t>(in.ctx);
    r.state_bytes = static_cast<uint64_t>(in.n_delta) * in.nv * in.dk * in.dv * 4ull +
                    static_cast<uint64_t>(in.n_delta) * in.conv_dim * in.conv_k * 4ull;
    r.need_bytes = in.weight_bytes + r.kv_bytes + r.state_bytes + in.pad_bytes;
    if (in.available_ram && r.need_bytes > in.available_ram) {
        r.ok = false;
        r.message = "OOM: need " + std::to_string(r.need_bytes) + " bytes, available " +
                    std::to_string(in.available_ram) + " (ctx=" + std::to_string(in.ctx) + ")";
    } else {
        r.ok = true;
        r.message = "ok";
    }
    return r;
}

} // namespace rapidllm
