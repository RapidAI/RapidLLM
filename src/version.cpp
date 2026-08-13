#include "rapidllm/version.h"

namespace rapidllm {
const char* version() {
#ifdef RAPIDLLM_VERSION
    return RAPIDLLM_VERSION;
#else
    return "0.1.0";
#endif
}
}
