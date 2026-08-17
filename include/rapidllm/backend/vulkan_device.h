#pragma once

#include "rapidllm/backend/device.h"

#include <memory>
#include <string>

namespace rapidllm {

// Creates a Vulkan 1.1 compute device (dynamic loader). Throws on missing ICD.
std::unique_ptr<Device> create_vulkan_device();

// True if the loader found a compute-capable physical device (does not create queues).
bool vulkan_available();

const char* vulkan_last_error();

} // namespace rapidllm
