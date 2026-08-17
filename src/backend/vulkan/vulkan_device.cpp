#include "rapidllm/backend/vulkan_device.h"
#include "rapidllm/backend/vulkan_spv.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Minimal Vulkan 1.1 declarations (no SDK required at compile time).
#if defined(_WIN32)
#define VKAPI_PTR __stdcall
#else
#define VKAPI_PTR
#endif

using VkFlags = uint32_t;
using VkBool32 = uint32_t;
using VkDeviceSize = uint64_t;
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkBuffer_T;
struct VkDeviceMemory_T;
struct VkShaderModule_T;
struct VkDescriptorSetLayout_T;
struct VkPipelineLayout_T;
struct VkPipeline_T;
struct VkDescriptorPool_T;
struct VkDescriptorSet_T;
struct VkCommandPool_T;
struct VkCommandBuffer_T;
struct VkFence_T;
using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkBuffer = VkBuffer_T*;
using VkDeviceMemory = VkDeviceMemory_T*;
using VkShaderModule = VkShaderModule_T*;
using VkDescriptorSetLayout = VkDescriptorSetLayout_T*;
using VkPipelineLayout = VkPipelineLayout_T*;
using VkPipeline = VkPipeline_T*;
using VkDescriptorPool = VkDescriptorPool_T*;
using VkDescriptorSet = VkDescriptorSet_T*;
using VkCommandPool = VkCommandPool_T*;
using VkCommandBuffer = VkCommandBuffer_T*;
using VkFence = VkFence_T*;
using VkResult = int32_t;

static constexpr VkResult VK_SUCCESS = 0;
static constexpr uint32_t VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
static constexpr uint32_t VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
static constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
static constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
static constexpr uint32_t VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
static constexpr uint32_t VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
static constexpr uint32_t VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
static constexpr uint32_t VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
static constexpr uint32_t VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
static constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
static constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
static constexpr uint32_t VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
static constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
static constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
static constexpr uint32_t VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
static constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
static constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
static constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
static constexpr uint32_t VK_QUEUE_COMPUTE_BIT = 2;
static constexpr uint32_t VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2;
static constexpr uint32_t VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4;
static constexpr uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20;
static constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x1;
static constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x2;
static constexpr uint32_t VK_SHARING_MODE_EXCLUSIVE = 0;
static constexpr uint32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
static constexpr uint32_t VK_SHADER_STAGE_COMPUTE_BIT = 0x20;
static constexpr uint32_t VK_PIPELINE_BIND_POINT_COMPUTE = 1;
static constexpr uint32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
static constexpr uint32_t VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 2;
static constexpr uint32_t VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 1;

struct VkApplicationInfo {
    uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};
struct VkInstanceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};
struct VkExtent3D {
    uint32_t width, height, depth;
};
struct VkQueueFamilyProperties {
    uint32_t queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
};
struct VkDeviceQueueCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
};
struct VkDeviceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};
struct VkMemoryType {
    uint32_t propertyFlags;
    uint32_t heapIndex;
};
struct VkMemoryHeap {
    VkDeviceSize size;
    uint32_t flags;
};
struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};
struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
};
struct VkMemoryAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
};
struct VkBufferCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    uint32_t usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};
struct VkShaderModuleCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
};
struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    uint32_t stageFlags;
    const void* pImmutableSamplers;
};
struct VkDescriptorSetLayoutCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};
struct VkPushConstantRange {
    uint32_t stageFlags;
    uint32_t offset;
    uint32_t size;
};
struct VkPipelineLayoutCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
};
struct VkPipelineShaderStageCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};
struct VkComputePipelineCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
};
struct VkDescriptorPoolSize {
    uint32_t type;
    uint32_t descriptorCount;
};
struct VkDescriptorPoolCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};
struct VkDescriptorSetAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};
struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};
struct VkWriteDescriptorSet {
    uint32_t sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};
struct VkCommandPoolCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
};
struct VkCommandBufferAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};
struct VkCommandBufferBeginInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
};
struct VkSubmitInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const uint32_t* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
};
struct VkFenceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
};
// Driver writes the full VkPhysicalDeviceProperties (~1 KiB). Keep a blob.
struct VkPhysicalDevicePropertiesBlob {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t rest[1536];
};

#define VK_FN(ret, name, ...) using PFN_##name = ret(VKAPI_PTR*)(__VA_ARGS__)

VK_FN(VkResult, vkCreateInstance, const VkInstanceCreateInfo*, const void*, VkInstance*);
VK_FN(void, vkDestroyInstance, VkInstance, const void*);
VK_FN(VkResult, vkEnumeratePhysicalDevices, VkInstance, uint32_t*, VkPhysicalDevice*);
VK_FN(void, vkGetPhysicalDeviceQueueFamilyProperties, VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
VK_FN(void, vkGetPhysicalDeviceMemoryProperties, VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
VK_FN(void, vkGetPhysicalDeviceProperties, VkPhysicalDevice, VkPhysicalDevicePropertiesBlob*);
VK_FN(VkResult, vkCreateDevice, VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
VK_FN(void, vkDestroyDevice, VkDevice, const void*);
VK_FN(void, vkGetDeviceQueue, VkDevice, uint32_t, uint32_t, VkQueue*);
VK_FN(VkResult, vkCreateBuffer, VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
VK_FN(void, vkDestroyBuffer, VkDevice, VkBuffer, const void*);
VK_FN(void, vkGetBufferMemoryRequirements, VkDevice, VkBuffer, VkMemoryRequirements*);
VK_FN(VkResult, vkAllocateMemory, VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
VK_FN(void, vkFreeMemory, VkDevice, VkDeviceMemory, const void*);
VK_FN(VkResult, vkBindBufferMemory, VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
VK_FN(VkResult, vkMapMemory, VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
VK_FN(void, vkUnmapMemory, VkDevice, VkDeviceMemory);
VK_FN(VkResult, vkCreateShaderModule, VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
VK_FN(void, vkDestroyShaderModule, VkDevice, VkShaderModule, const void*);
VK_FN(VkResult, vkCreateDescriptorSetLayout, VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*,
      VkDescriptorSetLayout*);
VK_FN(void, vkDestroyDescriptorSetLayout, VkDevice, VkDescriptorSetLayout, const void*);
VK_FN(VkResult, vkCreatePipelineLayout, VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
VK_FN(void, vkDestroyPipelineLayout, VkDevice, VkPipelineLayout, const void*);
VK_FN(VkResult, vkCreateComputePipelines, VkDevice, void*, uint32_t, const VkComputePipelineCreateInfo*, const void*,
      VkPipeline*);
VK_FN(void, vkDestroyPipeline, VkDevice, VkPipeline, const void*);
VK_FN(VkResult, vkCreateDescriptorPool, VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
VK_FN(void, vkDestroyDescriptorPool, VkDevice, VkDescriptorPool, const void*);
VK_FN(VkResult, vkAllocateDescriptorSets, VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
VK_FN(void, vkUpdateDescriptorSets, VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
VK_FN(VkResult, vkCreateCommandPool, VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
VK_FN(void, vkDestroyCommandPool, VkDevice, VkCommandPool, const void*);
VK_FN(VkResult, vkAllocateCommandBuffers, VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
VK_FN(VkResult, vkBeginCommandBuffer, VkCommandBuffer, const VkCommandBufferBeginInfo*);
VK_FN(VkResult, vkEndCommandBuffer, VkCommandBuffer);
VK_FN(void, vkCmdBindPipeline, VkCommandBuffer, uint32_t, VkPipeline);
VK_FN(void, vkCmdBindDescriptorSets, VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t,
      const VkDescriptorSet*, uint32_t, const uint32_t*);
VK_FN(void, vkCmdPushConstants, VkCommandBuffer, VkPipelineLayout, uint32_t, uint32_t, uint32_t, const void*);
VK_FN(void, vkCmdDispatch, VkCommandBuffer, uint32_t, uint32_t, uint32_t);
VK_FN(VkResult, vkQueueSubmit, VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
VK_FN(VkResult, vkQueueWaitIdle, VkQueue);
VK_FN(VkResult, vkCreateFence, VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
VK_FN(void, vkDestroyFence, VkDevice, VkFence, const void*);
VK_FN(VkResult, vkWaitForFences, VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
VK_FN(VkResult, vkResetFences, VkDevice, uint32_t, const VkFence*);
VK_FN(VkResult, vkResetCommandBuffer, VkCommandBuffer, VkFlags);

#undef VK_FN

namespace rapidllm {
namespace {

std::string g_vk_err;

#ifdef _WIN32
using Lib = HMODULE;
static void* load_lib() { return static_cast<void*>(LoadLibraryA("vulkan-1.dll")); }
static void* load_fn(void* h, const char* n) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h), n));
}
#else
using Lib = void*;
static void* load_lib() {
    void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return h;
}
static void* load_fn(void* h, const char* n) { return dlsym(h, n); }
#endif

#define LOAD(h, name)                                                                                                  \
    name = reinterpret_cast<PFN_##name>(load_fn(h, #name));                                                            \
    if (!name) throw std::runtime_error(std::string("vulkan missing ") + #name)

struct VkFns {
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkResetFences vkResetFences = nullptr;
    PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;

    void load(void* h) {
        LOAD(h, vkCreateInstance);
        LOAD(h, vkDestroyInstance);
        LOAD(h, vkEnumeratePhysicalDevices);
        LOAD(h, vkGetPhysicalDeviceQueueFamilyProperties);
        LOAD(h, vkGetPhysicalDeviceMemoryProperties);
        LOAD(h, vkGetPhysicalDeviceProperties);
        LOAD(h, vkCreateDevice);
        LOAD(h, vkDestroyDevice);
        LOAD(h, vkGetDeviceQueue);
        LOAD(h, vkCreateBuffer);
        LOAD(h, vkDestroyBuffer);
        LOAD(h, vkGetBufferMemoryRequirements);
        LOAD(h, vkAllocateMemory);
        LOAD(h, vkFreeMemory);
        LOAD(h, vkBindBufferMemory);
        LOAD(h, vkMapMemory);
        LOAD(h, vkUnmapMemory);
        LOAD(h, vkCreateShaderModule);
        LOAD(h, vkDestroyShaderModule);
        LOAD(h, vkCreateDescriptorSetLayout);
        LOAD(h, vkDestroyDescriptorSetLayout);
        LOAD(h, vkCreatePipelineLayout);
        LOAD(h, vkDestroyPipelineLayout);
        LOAD(h, vkCreateComputePipelines);
        LOAD(h, vkDestroyPipeline);
        LOAD(h, vkCreateDescriptorPool);
        LOAD(h, vkDestroyDescriptorPool);
        LOAD(h, vkAllocateDescriptorSets);
        LOAD(h, vkUpdateDescriptorSets);
        LOAD(h, vkCreateCommandPool);
        LOAD(h, vkDestroyCommandPool);
        LOAD(h, vkAllocateCommandBuffers);
        LOAD(h, vkBeginCommandBuffer);
        LOAD(h, vkEndCommandBuffer);
        LOAD(h, vkCmdBindPipeline);
        LOAD(h, vkCmdBindDescriptorSets);
        LOAD(h, vkCmdPushConstants);
        LOAD(h, vkCmdDispatch);
        LOAD(h, vkQueueSubmit);
        LOAD(h, vkQueueWaitIdle);
        LOAD(h, vkCreateFence);
        LOAD(h, vkDestroyFence);
        LOAD(h, vkWaitForFences);
        LOAD(h, vkResetFences);
        LOAD(h, vkResetCommandBuffer);
    }
};

#undef LOAD

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " vk=" + std::to_string(r));
}

uint32_t find_mem(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, uint32_t flags) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("vulkan: no host-visible coherent memory");
}

class VkBuf final : public Buffer {
public:
    VkFns* f = nullptr;
    VkDevice dev = nullptr;
    VkBuffer buf = nullptr;
    VkDeviceMemory mem = nullptr;
    void* map = nullptr;
    size_t n = 0;

    ~VkBuf() override {
        if (!f || !dev) return;
        if (map) f->vkUnmapMemory(dev, mem);
        if (buf) f->vkDestroyBuffer(dev, buf, nullptr);
        if (mem) f->vkFreeMemory(dev, mem, nullptr);
    }
    void* host_ptr() override { return map; }
    size_t bytes() const override { return n; }
};

class VkStream final : public Stream {
public:
    void begin_step(const char*) override {}
    void end_step() override {}
    void synchronize() override {}
};

class VulkanDevice final : public Device {
public:
    void* lib = nullptr;
    VkFns f{};
    VkInstance inst = nullptr;
    VkPhysicalDevice phys = nullptr;
    VkDevice dev = nullptr;
    VkQueue q = nullptr;
    uint32_t qfam = 0;
    VkPhysicalDeviceMemoryProperties memp{};
    char dname[256]{};
    VkDescriptorSetLayout dsl = nullptr;
    VkPipelineLayout pl_gemv = nullptr;
    VkPipelineLayout pl_rms = nullptr;
    VkPipeline pipe_gemv = nullptr;
    VkPipeline pipe_rms = nullptr;
    VkDescriptorPool pool = nullptr;
    VkCommandPool cpool = nullptr;
    VkCommandBuffer cmd = nullptr;
    VkFence fence = nullptr;
    VkDescriptorSet set_a = nullptr;
    VkDescriptorSet set_b = nullptr;

    VulkanDevice() { init(); }
    ~VulkanDevice() override { shutdown(); }

    DeviceKind kind() const override { return DeviceKind::Vulkan; }
    const char* name() const override { return dname[0] ? dname : "vulkan"; }

    std::unique_ptr<Buffer> allocate(const BufferDesc& d) override {
        auto b = std::make_unique<VkBuf>();
        make_buf(std::max<size_t>(d.bytes, 4), b.get());
        return b;
    }
    std::unique_ptr<Stream> create_stream() override { return std::make_unique<VkStream>(); }
    void memcpy(TensorView dst, TensorView src, Stream&) override {
        const size_t n = src.buf && dst.buf ? std::min(src.buf->bytes(), dst.buf->bytes()) : 0;
        if (n && dst.host_ptr() && src.host_ptr()) std::memcpy(dst.host_ptr(), src.host_ptr(), n);
    }
    bool has_kernel(const char* name) const override {
        return name && (std::strcmp(name, "gemv_f32") == 0 || std::strcmp(name, "rmsnorm") == 0);
    }

    void launch(const char* kernel, const TensorView* args, int nargs, const void* params, size_t params_bytes,
                Stream&) override {
        if (!kernel || !args) return;
        if (std::strcmp(kernel, "gemv_f32") == 0) {
            if (nargs < 3 || params_bytes < 8) throw std::runtime_error("gemv_f32 args");
            const int* mn = static_cast<const int*>(params);
            dispatch_gemv(args[0], args[1], args[2], mn[0], mn[1]);
            return;
        }
        if (std::strcmp(kernel, "rmsnorm") == 0) {
            if (nargs < 3 || params_bytes < 8) throw std::runtime_error("rmsnorm args");
            struct P {
                int n;
                float eps;
            };
            const P* p = static_cast<const P*>(params);
            dispatch_rms(args[0], args[1], args[2], p->n, p->eps);
            return;
        }
        throw std::runtime_error(std::string("unknown vulkan kernel ") + kernel);
    }

private:
    void make_buf(size_t bytes, VkBuf* b) {
        b->f = &f;
        b->dev = dev;
        b->n = bytes;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(f.vkCreateBuffer(dev, &bi, nullptr, &b->buf), "vkCreateBuffer");
        VkMemoryRequirements req{};
        f.vkGetBufferMemoryRequirements(dev, b->buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            find_mem(memp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(f.vkAllocateMemory(dev, &ai, nullptr, &b->mem), "vkAllocateMemory");
        check(f.vkBindBufferMemory(dev, b->buf, b->mem, 0), "vkBindBufferMemory");
        check(f.vkMapMemory(dev, b->mem, 0, req.size, 0, &b->map), "vkMapMemory");
        std::memset(b->map, 0, bytes);
    }

    VkBuf* as_buf(Buffer* b) {
        auto* v = dynamic_cast<VkBuf*>(b);
        if (!v) throw std::runtime_error("vulkan launch expects vulkan buffers");
        return v;
    }

    void write_set(VkDescriptorSet set, VkBuffer w, VkBuffer x, VkBuffer y) {
        VkDescriptorBufferInfo bi[3]{};
        bi[0] = {w, 0, ~0ull};
        bi[1] = {x, 0, ~0ull};
        bi[2] = {y, 0, ~0ull};
        VkWriteDescriptorSet wr[3]{};
        for (int i = 0; i < 3; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = set;
            wr[i].dstBinding = static_cast<uint32_t>(i);
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        f.vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);
    }

    void submit(VkPipeline pipe, VkPipelineLayout pl, VkDescriptorSet set, const void* pc, uint32_t pcs, uint32_t gx) {
        check(f.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check(f.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
        if (pc && pcs) f.vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcs, pc);
        f.vkCmdDispatch(cmd, gx, 1, 1);
        check(f.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        check(f.vkResetFences(dev, 1, &fence), "vkResetFences");
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        check(f.vkQueueSubmit(q, 1, &si, fence), "vkQueueSubmit");
        check(f.vkWaitForFences(dev, 1, &fence, 1, ~0ull), "vkWaitForFences");
    }

    void dispatch_gemv(const TensorView& W, const TensorView& X, const TensorView& Y, int m, int n) {
        if (m <= 0 || n <= 0) return;
        VkBuf* w = as_buf(W.buf);
        VkBuf* x = as_buf(X.buf);
        VkBuf* y = as_buf(Y.buf);
        write_set(set_a, w->buf, x->buf, y->buf);
        const int pc[2] = {m, n};
        submit(pipe_gemv, pl_gemv, set_a, pc, 8, static_cast<uint32_t>(m));
    }

    void dispatch_rms(const TensorView& X, const TensorView& G, const TensorView& Y, int n, float eps) {
        if (n <= 0) return;
        VkBuf* x = as_buf(X.buf);
        VkBuf* g = as_buf(G.buf);
        VkBuf* y = as_buf(Y.buf);
        write_set(set_b, x->buf, g->buf, y->buf);
        struct P {
            int n;
            float eps;
        } pc{n, eps};
        submit(pipe_rms, pl_rms, set_b, &pc, 8, 1);
    }

    VkShaderModule mk_shader(const uint32_t* words, size_t nwords) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = nwords * 4;
        ci.pCode = words;
        VkShaderModule sm = nullptr;
        check(f.vkCreateShaderModule(dev, &ci, nullptr, &sm), "vkCreateShaderModule");
        return sm;
    }

    void init() {
        lib = load_lib();
        if (!lib) throw std::runtime_error("vulkan loader not found (vulkan-1.dll / libvulkan.so.1)");
        f.load(lib);
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "RapidLLM";
        app.apiVersion = (1u << 22) | (1u << 12);  // 1.1
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        check(f.vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
        uint32_t ndev = 0;
        check(f.vkEnumeratePhysicalDevices(inst, &ndev, nullptr), "vkEnumeratePhysicalDevices");
        if (!ndev) throw std::runtime_error("vulkan: no physical devices");
        std::vector<VkPhysicalDevice> devs(ndev);
        check(f.vkEnumeratePhysicalDevices(inst, &ndev, devs.data()), "vkEnumeratePhysicalDevices");
        phys = nullptr;
        for (VkPhysicalDevice pd : devs) {
            uint32_t nq = 0;
            f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qf(nq);
            f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
            for (uint32_t i = 0; i < nq; ++i) {
                if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    phys = pd;
                    qfam = i;
                    break;
                }
            }
            if (phys) break;
        }
        if (!phys) throw std::runtime_error("vulkan: no compute queue");
        VkPhysicalDevicePropertiesBlob props{};
        f.vkGetPhysicalDeviceProperties(phys, &props);
        std::memcpy(dname, props.deviceName, 255);
        f.vkGetPhysicalDeviceMemoryProperties(phys, &memp);
        const float prio = 1.f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        check(f.vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
        f.vkGetDeviceQueue(dev, qfam, 0, &q);

        VkDescriptorSetLayoutBinding binds[3]{};
        for (int i = 0; i < 3; ++i) {
            binds[i].binding = static_cast<uint32_t>(i);
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 3;
        dlci.pBindings = binds;
        check(f.vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl), "vkCreateDescriptorSetLayout");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 8;
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        check(f.vkCreatePipelineLayout(dev, &plci, nullptr, &pl_gemv), "vkCreatePipelineLayout gemv");
        check(f.vkCreatePipelineLayout(dev, &plci, nullptr, &pl_rms), "vkCreatePipelineLayout rms");

        VkShaderModule sm_g = mk_shader(kSpvGemvF32, sizeof(kSpvGemvF32) / 4);
        VkShaderModule sm_r = mk_shader(kSpvRmsNorm, sizeof(kSpvRmsNorm) / 4);
        auto mk_pipe = [&](VkShaderModule sm, VkPipelineLayout pl, VkPipeline* out) {
            VkComputePipelineCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            ci.stage.sType = 18;  // PIPELINE_SHADER_STAGE_CREATE_INFO
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = sm;
            ci.stage.pName = "main";
            ci.layout = pl;
            ci.basePipelineIndex = -1;
            check(f.vkCreateComputePipelines(dev, nullptr, 1, &ci, nullptr, out), "vkCreateComputePipelines");
        };
        mk_pipe(sm_g, pl_gemv, &pipe_gemv);
        mk_pipe(sm_r, pl_rms, &pipe_rms);
        f.vkDestroyShaderModule(dev, sm_g, nullptr);
        f.vkDestroyShaderModule(dev, sm_r, nullptr);

        VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = 8;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psz;
        check(f.vkCreateDescriptorPool(dev, &dpci, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetLayout layouts[2] = {dsl, dsl};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 2;
        dai.pSetLayouts = layouts;
        VkDescriptorSet sets[2]{};
        check(f.vkAllocateDescriptorSets(dev, &dai, sets), "vkAllocateDescriptorSets");
        set_a = sets[0];
        set_b = sets[1];

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qfam;
        check(f.vkCreateCommandPool(dev, &cpci, nullptr, &cpool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(f.vkAllocateCommandBuffers(dev, &cai, &cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(f.vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence");
    }

    void shutdown() {
        if (!dev) {
            if (inst) f.vkDestroyInstance(inst, nullptr);
            return;
        }
        f.vkQueueWaitIdle(q);
        if (fence) f.vkDestroyFence(dev, fence, nullptr);
        if (cpool) f.vkDestroyCommandPool(dev, cpool, nullptr);
        if (pool) f.vkDestroyDescriptorPool(dev, pool, nullptr);
        if (pipe_gemv) f.vkDestroyPipeline(dev, pipe_gemv, nullptr);
        if (pipe_rms) f.vkDestroyPipeline(dev, pipe_rms, nullptr);
        if (pl_gemv) f.vkDestroyPipelineLayout(dev, pl_gemv, nullptr);
        if (pl_rms) f.vkDestroyPipelineLayout(dev, pl_rms, nullptr);
        if (dsl) f.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        f.vkDestroyDevice(dev, nullptr);
        if (inst) f.vkDestroyInstance(inst, nullptr);
    }
};

} // namespace

bool vulkan_available() {
    try {
        void* h = load_lib();
        if (!h) {
            g_vk_err = "loader not found";
            return false;
        }
        auto enum_pd = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(load_fn(h, "vkEnumeratePhysicalDevices"));
        auto create = reinterpret_cast<PFN_vkCreateInstance>(load_fn(h, "vkCreateInstance"));
        auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(load_fn(h, "vkDestroyInstance"));
        if (!enum_pd || !create || !destroy) {
            g_vk_err = "core fns missing";
            return false;
        }
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "RapidLLM";
        app.apiVersion = (1u << 22);
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VkInstance inst = nullptr;
        if (create(&ici, nullptr, &inst) != VK_SUCCESS) {
            g_vk_err = "vkCreateInstance failed";
            return false;
        }
        uint32_t n = 0;
        const VkResult r = enum_pd(inst, &n, nullptr);
        destroy(inst, nullptr);
        if (r != VK_SUCCESS || n == 0) {
            g_vk_err = "no physical devices";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        g_vk_err = e.what();
        return false;
    }
}

const char* vulkan_last_error() { return g_vk_err.empty() ? "" : g_vk_err.c_str(); }

std::unique_ptr<Device> create_vulkan_device() {
    try {
        auto d = std::make_unique<VulkanDevice>();
        g_vk_err.clear();
        return d;
    } catch (const std::exception& e) {
        g_vk_err = e.what();
        throw;
    }
}

} // namespace rapidllm
