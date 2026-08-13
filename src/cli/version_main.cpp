#include "rapidllm/version.h"
#include "rapidllm/api.h"

#include <cstdio>

int main() {
    std::printf("rapidllm %s api=%d\n", rapidllm_version_string(), rapidllm_api_version());
    return 0;
}
