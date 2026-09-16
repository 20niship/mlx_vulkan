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

#include <mkx/core/op_node.hpp>

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

  // begin_replay/end_replayで挟んだdispatch()群を1本のVkCommandBufferとしてkeyで保持し、以後replay(key)のみで再submitできるようにする(グラフ再構築を回避)。
  static bool has_replay(const void* key);
  static void begin_replay(const void* key);
  static void end_replay(const void* key);
  static void replay(const void* key);
  static void invalidate_replay(const void* key);

  // Live allocator/pool stats (buffer count + bytes currently retained), for leak diagnostics. Cheap; safe to poll between benchmark iterations.
  static std::string debug_stats();

  // node自身のアドレスをキーに初回alloc・以降は同じBufferを返す(is_permanentなら(loc_id,owner)キーの永続マップに委譲)。
  static Buffer* get_or_allocate(const OpNode<VulkanBackend>* node, size_t nbytes);
  // get_or_allocateと違い新規allocは行わない。既にバッファが実体化済みかどうかだけを調べる(テスト用)。
  static bool has_buffer(const OpNode<VulkanBackend>* node);
  // NodePtrのカスタムdeleterから呼ばれる。transientなら対応するBufferをfree、permanentなら何もしない。
  static void release_node(const OpNode<VulkanBackend>* node);
  // owner一致の永続バッファを全てfreeする。
  static void release_persistent_for_owner(const void* owner);
};

} // namespace mkx
