#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <vulkan/vulkan.h>

#ifdef MKX_USE_VMA
#include <vk_mem_alloc.h>
#endif

namespace mkx {

struct VulkanBackend {
  struct Buffer {
    VkBuffer buffer       = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size           = 0;
#ifdef MKX_USE_VMA
    VmaAllocation allocation = VK_NULL_HANDLE;
#endif
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

  // 解除し忘れて残ったバッファをContextデストラクタで強制解放するための安全網登録。
  static void register_persistent(Buffer* buf);
  static void unregister_persistent(Buffer* buf);

  // Live allocator/pool stats (buffer count + bytes currently retained), for leak diagnostics. Cheap; safe to poll between benchmark iterations.
  static std::string debug_stats();
};

} // namespace mkx
