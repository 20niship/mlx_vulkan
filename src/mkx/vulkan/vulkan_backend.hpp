#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <vulkan/vulkan.h>

namespace mkx {

struct VulkanBackend {
  struct Buffer {
    VkBuffer buffer       = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size           = 0;
  };

  struct Pipeline {
    VkPipeline pipeline              = VK_NULL_HANDLE;
    VkPipelineLayout layout          = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkShaderModule module            = VK_NULL_HANDLE;
    uint32_t binding_count           = 0;
  };

  static Pipeline compile(std::string_view source, size_t hash);
  static Buffer* alloc(size_t nbytes);
  static void free(Buffer* buf);
  static void upload(Buffer* buf, const void* data, size_t nbytes);
  static void download(Buffer* buf, void* data, size_t nbytes);
  static void dispatch(Pipeline& pipeline, std::span<Buffer*> buffers, std::span<const std::byte> push_data, std::array<uint32_t, 3> groups);
  static void wait_idle();
};

} // namespace mkx
