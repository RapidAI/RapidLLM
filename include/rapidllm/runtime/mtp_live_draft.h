#pragma once

namespace rapidllm {

// Timed draft slots must equal MTP kernel launches. memcpy-pad is not live MTP.
inline int live_draft_count(int n_mtp_launches) {
    return n_mtp_launches > 0 ? n_mtp_launches : 0;
}

}  // namespace rapidllm
