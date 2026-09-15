#ifdef MKX_USE_VMA
#define VMA_IMPLEMENTATION
#endif
#include <mkx/vulkan/vulkan_backend.hpp>

#include <mkx/shaders/shader_source.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mkx {

namespace {

#ifdef MKX_ENABLE_VALIDATION
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT /*type*/, const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user_data*/) {
  const char* tag = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARN" : "INFO";
  fprintf(stderr, "[mkx][validation][%s] %s\n", tag, data->pMessage);
  return VK_FALSE;
}
#endif

struct Context {
  VkInstance instance        = VK_NULL_HANDLE;
  VkPhysicalDevice physical  = VK_NULL_HANDLE;
  VkDevice device            = VK_NULL_HANDLE;
  VkQueue queue              = VK_NULL_HANDLE;
  uint32_t queue_family      = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;
#ifdef MKX_ENABLE_VALIDATION
  VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
#endif
#ifdef MKX_USE_VMA
  VmaAllocator allocator = VK_NULL_HANDLE;
#endif
  std::unordered_set<VulkanBackend::Buffer*> persistent_bufs;

  Context() {
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "mkx";
    app_info.apiVersion       = VK_API_VERSION_1_3;

    uint32_t inst_ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, nullptr);
    std::vector<VkExtensionProperties> inst_exts(inst_ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, inst_exts.data());

    // MoltenVKはVK_KHR_portability_enumeration必須、Linux+lavapipe等には無いため存在する場合のみ有効化する。
    const char* portability_enum_ext = "VK_KHR_portability_enumeration";
    bool has_portability_enum        = false;
    for(auto& e : inst_exts) {
      if(std::string_view(e.extensionName) == portability_enum_ext) has_portability_enum = true;
    }

    std::vector<const char*> inst_extensions;
    if(has_portability_enum) inst_extensions.push_back(portability_enum_ext);

    std::vector<const char*> inst_layers;
#ifdef MKX_ENABLE_VALIDATION
    // 診断ビルドのみ: レイヤ自体の有無を確認してから有効化(未インストール環境でvkCreateInstanceを失敗させない)。
    bool has_debug_utils = false;
    for(auto& e : inst_exts) {
      if(std::string_view(e.extensionName) == VK_EXT_DEBUG_UTILS_EXTENSION_NAME) has_debug_utils = true;
    }
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    bool has_validation_layer = false;
    for(auto& l : layers) {
      if(std::string_view(l.layerName) == "VK_LAYER_KHRONOS_validation") has_validation_layer = true;
    }
    if(has_validation_layer && has_debug_utils) {
      inst_layers.push_back("VK_LAYER_KHRONOS_validation");
      inst_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else {
      fprintf(stderr, "[mkx] MKX_ENABLE_VALIDATION requested but VK_LAYER_KHRONOS_validation/VK_EXT_debug_utils not available; skipping.\n");
    }
#endif

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo        = &app_info;
    instance_info.enabledExtensionCount   = static_cast<uint32_t>(inst_extensions.size());
    instance_info.ppEnabledExtensionNames = inst_extensions.data();
    instance_info.enabledLayerCount       = static_cast<uint32_t>(inst_layers.size());
    instance_info.ppEnabledLayerNames     = inst_layers.data();
    if(has_portability_enum) instance_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

#ifdef MKX_ENABLE_VALIDATION
    VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_info.pfnUserCallback = debug_callback;
    if(!inst_layers.empty()) instance_info.pNext = &messenger_info; // instance生成時点からのメッセージも捕捉する
#endif

    if(vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("mkx: vkCreateInstance failed");
    }

#ifdef MKX_ENABLE_VALIDATION
    if(!inst_layers.empty()) {
      auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
      if(create_fn) create_fn(instance, &messenger_info, nullptr, &debug_messenger);
    }
#endif

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if(device_count == 0) throw std::runtime_error("mkx: no Vulkan physical device found");
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    physical = devices[0];

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    queue_family = 0;
    for(uint32_t i = 0; i < family_count; ++i) {
      if(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queue_family = i;
        break;
      }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount       = 1;
    queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos    = &queue_info;

    const char* portability_ext = "VK_KHR_portability_subset";
    uint32_t ext_count          = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, exts.data());
    std::vector<const char*> enabled_exts;
    for(auto& e : exts) {
      if(std::string_view(e.extensionName) == portability_ext) enabled_exts.push_back(portability_ext);
    }
    device_info.enabledExtensionCount   = static_cast<uint32_t>(enabled_exts.size());
    device_info.ppEnabledExtensionNames = enabled_exts.data();

    if(vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) {
      throw std::runtime_error("mkx: vkCreateDevice failed");
    }
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);

#ifdef MKX_USE_VMA
    VmaAllocatorCreateInfo vma_info{};
    vma_info.physicalDevice   = physical;
    vma_info.device           = device;
    vma_info.instance         = instance;
    vma_info.vulkanApiVersion = VK_API_VERSION_1_3;
    if(vmaCreateAllocator(&vma_info, &allocator) != VK_SUCCESS) {
      throw std::runtime_error("mkx: vmaCreateAllocator failed");
    }
#endif
  }

  // 破棄しないとvalidation layerのobject-leak検出(プロセス終了時のvkDestroyInstance相当)が機能しない。
  ~Context() {
    if(device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(device);
    // release_persistent_for_ownerを呼び忘れたまま残ったバッファの安全網。
    for(auto* buf : persistent_bufs) {
#ifdef MKX_USE_VMA
      vmaDestroyBuffer(allocator, buf->buffer, buf->allocation);
#else
      vkDestroyBuffer(device, buf->buffer, nullptr);
      vkFreeMemory(device, buf->memory, nullptr);
#endif
      delete buf;
    }
    persistent_bufs.clear();
#ifdef MKX_USE_VMA
    if(allocator) vmaDestroyAllocator(allocator);
#endif
    if(command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDevice(device, nullptr);
#ifdef MKX_ENABLE_VALIDATION
    if(debug_messenger) {
      auto destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
      if(destroy_fn) destroy_fn(instance, debug_messenger, nullptr);
    }
#endif
    if(instance) vkDestroyInstance(instance, nullptr);
  }
};

Context& ctx() {
  static Context context;
  return context;
}

// 旧実装はdispatch()ごとにcommand buffer alloc+submit+vkQueueWaitIdleしており、1 evalあたりの実dispatch数に比例したCPU側オーバーヘッドが支配的だった。同一eval()内のdispatchをここに積み、次のwait_idle()で1回のsubmit+waitにまとめる(descriptor poolは実行完了までpendingに保持)。
struct Batch {
  VkCommandBuffer cmd     = VK_NULL_HANDLE;
  bool open               = false;
  bool has_prior_dispatch = false;
  std::vector<VkDescriptorPool> pending_desc_pools;
  // 容量を使い切るまで使い回すpool(vkCreate/DestroyDescriptorPoolの呼び出し回数を減らす。尽きたら新しいpoolをpending_desc_poolsに追加)。
  VkDescriptorPool current_pool                  = VK_NULL_HANDLE;
  uint32_t current_pool_sets_used                = 0;
  static constexpr uint32_t kPoolSetCapacity     = 64;
  static constexpr uint32_t kPoolBindingCapacity = 4096;
};

Batch& batch() {
  static Batch b;
  return b;
}

// 永続バッファはstep間でBuffer*が同一のまま渡されるため(pipeline,buffer列)キーでdescriptor setをLRUキャッシュし再作成を省く。transientはmiss前提でno-op相当。
struct DescCache {
  static constexpr uint32_t kCapacity = 2048;
  VkDescriptorPool pool               = VK_NULL_HANDLE;
  std::unordered_map<uint64_t, VkDescriptorSet> map;
  std::list<uint64_t> lru; // front=most recently used
  std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru_pos;
};

DescCache& desc_cache() {
  static DescCache dc;
  return dc;
}

void ensure_desc_cache_pool() {
  auto& dc = desc_cache();
  if(dc.pool != VK_NULL_HANDLE) return;
  auto& c = ctx();
  VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, DescCache::kCapacity * 8};
  VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = DescCache::kCapacity;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes    = &pool_size;
  vkCreateDescriptorPool(c.device, &pool_info, nullptr, &dc.pool);
}

uint64_t hash_dispatch_key(VkPipeline pipeline, std::span<VulkanBackend::Buffer*> buffers) {
  uint64_t h = std::hash<void*>{}(reinterpret_cast<void*>(pipeline));
  for(auto* b : buffers) {
    h ^= std::hash<void*>{}(b) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  return h;
}

// キャッシュヒット時はVkDescriptorSetを返す。ミス時はnullptrを返す(呼び出し側が新規作成しcache_insertでここへ登録する)。
VkDescriptorSet desc_cache_lookup(uint64_t key) {
  auto& dc = desc_cache();
  auto it  = dc.map.find(key);
  if(it == dc.map.end()) return VK_NULL_HANDLE;
  dc.lru.erase(dc.lru_pos.at(key));
  dc.lru.push_front(key);
  dc.lru_pos[key] = dc.lru.begin();
  return it->second;
}

void desc_cache_insert(uint64_t key, VkDescriptorSet set) {
  auto& c  = ctx();
  auto& dc = desc_cache();
  if(dc.map.size() >= DescCache::kCapacity) {
    uint64_t evict_key = dc.lru.back();
    dc.lru.pop_back();
    dc.lru_pos.erase(evict_key);
    VkDescriptorSet evict_set = dc.map.at(evict_key);
    dc.map.erase(evict_key);
    vkFreeDescriptorSets(c.device, dc.pool, 1, &evict_set);
  }
  dc.map[key] = set;
  dc.lru.push_front(key);
  dc.lru_pos[key] = dc.lru.begin();
}

void flush_batch() {
  auto& b = batch();
  if(!b.open) return;
  auto& c = ctx();
  vkEndCommandBuffer(b.cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &b.cmd;
  vkQueueSubmit(c.queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(c.queue);

  vkFreeCommandBuffers(c.device, c.command_pool, 1, &b.cmd);
  for(auto pool : b.pending_desc_pools) vkDestroyDescriptorPool(c.device, pool, nullptr);
  b.pending_desc_pools.clear();
  b.cmd                    = VK_NULL_HANDLE;
  b.open                   = false;
  b.has_prior_dispatch     = false;
  b.current_pool           = VK_NULL_HANDLE;
  b.current_pool_sets_used = 0;
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
  for(uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  throw std::runtime_error("mkx: no suitable Vulkan memory type");
}

std::vector<uint32_t> compile_glsl_to_spirv(std::string_view source) {
  char src_path[] = "/tmp/mkx_shader_XXXXXX.comp";
  int fd          = mkstemps(src_path, 5);
  if(fd < 0) throw std::runtime_error("mkx: mkstemps failed");
  {
    std::ofstream out(src_path, std::ios::trunc);
    out << source;
  }
  close(fd);

  std::string spv_path = std::string(src_path) + ".spv";
  std::string cmd      = "glslangValidator -V --target-env vulkan1.2 -S comp -o " + spv_path + " " + src_path + " >/tmp/mkx_glslang.log 2>&1";
  int rc               = std::system(cmd.c_str());
  if(rc != 0) {
    std::ifstream log(spv_path.substr(0, spv_path.size() - 4) + ".log");
    throw std::runtime_error("mkx: glslangValidator failed compiling shader");
  }

  std::ifstream spv(spv_path, std::ios::binary | std::ios::ate);
  if(!spv) throw std::runtime_error("mkx: failed to read compiled SPIR-V");
  size_t size = static_cast<size_t>(spv.tellg());
  spv.seekg(0);
  std::vector<uint32_t> code(size / sizeof(uint32_t));
  spv.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

  std::remove(src_path);
  std::remove(spv_path.c_str());
  return code;
}

} // namespace

VulkanBackend::Pipeline VulkanBackend::compile(std::string_view source, size_t /*hash*/) {
  auto& c    = ctx();
  auto spirv = compile_glsl_to_spirv(source);

  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv.size() * sizeof(uint32_t);
  module_info.pCode    = spirv.data();

  Pipeline pipe;
  if(vkCreateShaderModule(c.device, &module_info, nullptr, &pipe.module) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkCreateShaderModule failed");
  }

  // バッファ数はshader source中の "binding = N" の最大値+1から数える(全shaderはbinding 0..k連番)
  uint32_t max_binding = 0;
  size_t pos           = 0;
  while((pos = source.find("binding = ", pos)) != std::string_view::npos) {
    pos += 10;
    uint32_t b = static_cast<uint32_t>(std::strtoul(source.data() + pos, nullptr, 10));
    if(b + 1 > max_binding) max_binding = b + 1;
  }
  pipe.binding_count = max_binding;

  std::vector<VkDescriptorSetLayoutBinding> bindings(max_binding);
  for(uint32_t i = 0; i < max_binding; ++i) {
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  set_layout_info.bindingCount = max_binding;
  set_layout_info.pBindings    = bindings.data();
  vkCreateDescriptorSetLayout(c.device, &set_layout_info, nullptr, &pipe.set_layout);

  VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount         = 1;
  layout_info.pSetLayouts            = &pipe.set_layout;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges    = &push_range;
  vkCreatePipelineLayout(c.device, &layout_info, nullptr, &pipe.layout);

  VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = pipe.module;
  stage_info.pName  = "main";

  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage  = stage_info;
  pipeline_info.layout = pipe.layout;
  if(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipe.pipeline) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkCreateComputePipelines failed");
  }
  return pipe;
}

// OpNodeは1ステップごとに新規でalloc/freeされる(gpu_bufferはeval()単位で使い捨て)ため、サイズ別free-listでVkBuffer/VkDeviceMemory実体を使い回しvkAllocateMemory/vkCreateBufferの呼び出し頻度を下げる。1サイズあたりkMaxPooledPerSizeを超える分は素直に破棄する。
constexpr size_t kMaxPooledPerSize = 64;
// バケット横断のグローバルcap(安全弁)。実測ではcapを2048→256まで絞ってもwired page数の暴れと長時間学習でのクラッシュ挙動は変わらず、原因はこのプールではなくMoltenVK/Vulkanドライバ側のメモリ挙動と判断(詳細はdocs/perf-issuesの記録を参照)。
constexpr size_t kMaxPooledTotal = 1024;
size_t g_pooled_total            = 0;

std::unordered_map<size_t, std::vector<VulkanBackend::Buffer*>>& free_list() {
  static std::unordered_map<size_t, std::vector<VulkanBackend::Buffer*>> pool;
  return pool;
}

// free()単体でwait_idle()しない: 破棄/プール返却は次のwait_idle()(vkDeviceWaitIdle後で安全)にまとめて遅延する。
std::vector<VulkanBackend::Buffer*>& pending_frees() {
  static std::vector<VulkanBackend::Buffer*> q;
  return q;
}
void free_now(VulkanBackend::Buffer* buf);

#ifdef MKX_USE_VMA

// VMAが自前でブロック単位に再利用するため、kMaxPooledPerSize/kMaxPooledTotalの自作プールは不要(free()は素直にvmaDestroyBuffer)。

VulkanBackend::Buffer* VulkanBackend::alloc(size_t nbytes) {
  auto& c   = ctx();
  auto* buf = new Buffer();
  buf->size = nbytes;

  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size        = nbytes;
  buffer_info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

  if(vmaCreateBuffer(c.allocator, &buffer_info, &alloc_info, &buf->buffer, &buf->allocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vmaCreateBuffer failed (nbytes=" + std::to_string(nbytes) + ")");
  }
  return buf;
}

void VulkanBackend::free(Buffer* buf) {
  if(!buf) return;
  pending_frees().push_back(buf);
}

void free_now(VulkanBackend::Buffer* buf) {
  vmaDestroyBuffer(ctx().allocator, buf->buffer, buf->allocation);
  delete buf;
}

void VulkanBackend::upload(Buffer* buf, const void* data, size_t nbytes) {
  auto& c      = ctx();
  void* mapped = nullptr;
  if(vmaMapMemory(c.allocator, buf->allocation, &mapped) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vmaMapMemory failed in upload (nbytes=" + std::to_string(nbytes) + ")");
  }
  std::memcpy(mapped, data, nbytes);
  if(vmaFlushAllocation(c.allocator, buf->allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
    vmaUnmapMemory(c.allocator, buf->allocation);
    throw std::runtime_error("mkx: vmaFlushAllocation failed in upload (nbytes=" + std::to_string(nbytes) + ")");
  }
  vmaUnmapMemory(c.allocator, buf->allocation);
}

void VulkanBackend::download(Buffer* buf, void* data, size_t nbytes) {
  flush_batch();
  auto& c      = ctx();
  void* mapped = nullptr;
  if(vmaMapMemory(c.allocator, buf->allocation, &mapped) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vmaMapMemory failed in download (nbytes=" + std::to_string(nbytes) + ")");
  }
  std::memcpy(data, mapped, nbytes);
  vmaUnmapMemory(c.allocator, buf->allocation);
}

std::string VulkanBackend::debug_stats() {
  char* json = nullptr;
  vmaBuildStatsString(ctx().allocator, &json, VK_TRUE);
  std::string result = json ? json : "";
  vmaFreeStatsString(ctx().allocator, json);
  return result;
}

#else

VulkanBackend::Buffer* VulkanBackend::alloc(size_t nbytes) {
  auto& pool = free_list();
  auto it    = pool.find(nbytes);
  if(it != pool.end() && !it->second.empty()) {
    auto* buf = it->second.back();
    it->second.pop_back();
    g_pooled_total--;
    return buf;
  }

  auto& c   = ctx();
  auto* buf = new Buffer();
  buf->size = nbytes;

  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size        = nbytes;
  buffer_info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if(vkCreateBuffer(c.device, &buffer_info, nullptr, &buf->buffer) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkCreateBuffer failed (nbytes=" + std::to_string(nbytes) + ")");
  }

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(c.device, buf->buffer, &reqs);
  VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.allocationSize  = reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(c.physical, reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  // Silent failures here previously left buf->memory as VK_NULL_HANDLE; later vkMapMemory on it fails too (also unchecked), and the ensuing memcpy into a garbage pointer corrupts state until the OS kills the process -- surface the real error instead.
  if(vkAllocateMemory(c.device, &alloc_info, nullptr, &buf->memory) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkAllocateMemory failed (size=" + std::to_string(reqs.size) + ") -- likely too many live VkDeviceMemory allocations");
  }
  vkBindBufferMemory(c.device, buf->buffer, buf->memory, 0);
  return buf;
}

void VulkanBackend::free(Buffer* buf) {
  if(!buf) return;
  pending_frees().push_back(buf);
}

void free_now(VulkanBackend::Buffer* buf) {
  auto& pool = free_list();
  auto& slot = pool[buf->size];
  if(slot.size() < kMaxPooledPerSize && g_pooled_total < kMaxPooledTotal) {
    slot.push_back(buf);
    g_pooled_total++;
    return;
  }
  auto& c = ctx();
  vkDestroyBuffer(c.device, buf->buffer, nullptr);
  vkFreeMemory(c.device, buf->memory, nullptr);
  delete buf;
}

void VulkanBackend::upload(Buffer* buf, const void* data, size_t nbytes) {
  auto& c      = ctx();
  void* mapped = nullptr;
  if(vkMapMemory(c.device, buf->memory, 0, nbytes, 0, &mapped) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkMapMemory failed in upload (nbytes=" + std::to_string(nbytes) + ")");
  }
  std::memcpy(mapped, data, nbytes);
  vkUnmapMemory(c.device, buf->memory);
}

void VulkanBackend::download(Buffer* buf, void* data, size_t nbytes) {
  // A dispatch may still be sitting unsubmitted in the open batch; flush so this read sees its output.
  flush_batch();
  auto& c      = ctx();
  void* mapped = nullptr;
  if(vkMapMemory(c.device, buf->memory, 0, nbytes, 0, &mapped) != VK_SUCCESS) {
    throw std::runtime_error("mkx: vkMapMemory failed in download (nbytes=" + std::to_string(nbytes) + ")");
  }
  std::memcpy(data, mapped, nbytes);
  vkUnmapMemory(c.device, buf->memory);
}

std::string VulkanBackend::debug_stats() {
  size_t count = 0, bytes = 0;
  for(auto& [size, bufs] : free_list()) {
    count += bufs.size();
    bytes += size * bufs.size();
  }
  std::ostringstream os;
  os << "mkx pool: " << count << " buffers pooled (cap=" << kMaxPooledTotal << "), " << (bytes / (1024.0 * 1024.0)) << " MiB retained across " << free_list().size() << " distinct sizes";
  return os.str();
}

#endif

void VulkanBackend::dispatch(Pipeline& pipeline, std::span<Buffer*> buffers, std::span<const std::byte> push_data, std::array<uint32_t, 3> groups) {
  auto& c = ctx();
  auto& b = batch();

  if(!b.open) {
    VkCommandBufferAllocateInfo cmd_alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_alloc.commandPool        = c.command_pool;
    cmd_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(c.device, &cmd_alloc, &b.cmd);

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(b.cmd, &begin_info);
    b.open               = true;
    b.has_prior_dispatch = false;
  }

  uint64_t desc_key        = hash_dispatch_key(pipeline.pipeline, buffers);
  VkDescriptorSet desc_set = desc_cache_lookup(desc_key);
  if(desc_set == VK_NULL_HANDLE) {
    if(b.current_pool == VK_NULL_HANDLE || b.current_pool_sets_used >= Batch::kPoolSetCapacity) {
      VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Batch::kPoolBindingCapacity};
      VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool_info.maxSets       = Batch::kPoolSetCapacity;
      pool_info.poolSizeCount = 1;
      pool_info.pPoolSizes    = &pool_size;
      vkCreateDescriptorPool(c.device, &pool_info, nullptr, &b.current_pool);
      b.pending_desc_pools.push_back(b.current_pool);
      b.current_pool_sets_used = 0;
    }

    ensure_desc_cache_pool();
    auto& dc = desc_cache();
    VkDescriptorSetAllocateInfo cache_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    cache_alloc.descriptorPool     = dc.pool;
    cache_alloc.descriptorSetCount = 1;
    cache_alloc.pSetLayouts        = &pipeline.set_layout;
    if(vkAllocateDescriptorSets(c.device, &cache_alloc, &desc_set) != VK_SUCCESS) {
      // 永続pool枯渇(想定外の多様な組み合わせ)。このdispatchだけbatch局所poolへフォールバック、キャッシュには載せない。
      VkDescriptorSetAllocateInfo set_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      set_alloc.descriptorPool     = b.current_pool;
      set_alloc.descriptorSetCount = 1;
      set_alloc.pSetLayouts        = &pipeline.set_layout;
      vkAllocateDescriptorSets(c.device, &set_alloc, &desc_set);
      b.current_pool_sets_used++;
      desc_key = 0; // insertしない印
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for(size_t i = 0; i < buffers.size(); ++i) {
      buffer_infos[i]           = {buffers[i]->buffer, 0, VK_WHOLE_SIZE};
      writes[i]                 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet          = desc_set;
      writes[i].dstBinding      = static_cast<uint32_t>(i);
      writes[i].descriptorCount = 1;
      writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo     = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(c.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if(desc_key != 0) desc_cache_insert(desc_key, desc_set);
  }

  // Chained dispatches commonly read a prior dispatch's output via the same storage buffers; a full shader-write/read barrier avoids per-dispatch buffer-aliasing analysis.
  if(b.has_prior_dispatch) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(b.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  vkCmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
  vkCmdBindDescriptorSets(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &desc_set, 0, nullptr);
  if(!push_data.empty()) {
    vkCmdPushConstants(b.cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_data.size()), push_data.data());
  }
  vkCmdDispatch(b.cmd, groups[0], groups[1], groups[2]);
  b.has_prior_dispatch = true;
}

void VulkanBackend::wait_idle() {
  flush_batch();
  vkDeviceWaitIdle(ctx().device);
  auto& q = pending_frees();
  for(auto* buf : q) free_now(buf);
  q.clear();
}

namespace {

std::unordered_map<const void*, VulkanBackend::Buffer*>& transient_map() {
  static std::unordered_map<const void*, VulkanBackend::Buffer*> m;
  return m;
}

struct PersistentKeyHash {
  size_t operator()(const std::pair<uint64_t, const void*>& k) const { return std::hash<uint64_t>{}(k.first) ^ (std::hash<const void*>{}(k.second) << 1); }
};

std::unordered_map<std::pair<uint64_t, const void*>, VulkanBackend::Buffer*, PersistentKeyHash>& permanent_map() {
  static std::unordered_map<std::pair<uint64_t, const void*>, VulkanBackend::Buffer*, PersistentKeyHash> m;
  return m;
}

} // namespace

VulkanBackend::Buffer* VulkanBackend::get_or_allocate(const OpNode<VulkanBackend>* node, size_t nbytes) {
  if(node->is_permanent) {
    auto& pmap = permanent_map();
    auto key   = std::make_pair(node->persistent_loc_id, node->persistent_owner);
    auto it    = pmap.find(key);
    if(it != pmap.end()) {
      if(it->second->size == nbytes) return it->second;
      // 実shapeが変わった(衝突contact数の変動等)ので古いバッファを解放し新サイズで作り直す。
      ctx().persistent_bufs.erase(it->second);
      free(it->second);
      auto* resized = alloc(nbytes);
      ctx().persistent_bufs.insert(resized);
      it->second = resized;
      return resized;
    }
    auto* buf = alloc(nbytes);
    ctx().persistent_bufs.insert(buf); // 呼び忘れ安全網(~Context参照)への登録
    pmap.emplace(key, buf);
    return buf;
  }
  auto& tmap = transient_map();
  auto it    = tmap.find(node);
  if(it != tmap.end()) return it->second;
  auto* buf = alloc(nbytes);
  tmap.emplace(node, buf);
  return buf;
}

bool VulkanBackend::has_buffer(const OpNode<VulkanBackend>* node) {
  if(node->is_permanent) return permanent_map().count(std::make_pair(node->persistent_loc_id, node->persistent_owner)) > 0;
  return transient_map().count(node) > 0;
}

void VulkanBackend::release_node(const OpNode<VulkanBackend>* node) {
  if(node->is_permanent) return; // permanentの解放はrelease_persistent_for_ownerに一任
  auto& tmap = transient_map();
  auto it    = tmap.find(node);
  if(it == tmap.end()) return;
  free(it->second);
  tmap.erase(it);
}

void VulkanBackend::release_persistent_for_owner(const void* owner) {
  auto& pmap = permanent_map();
  for(auto it = pmap.begin(); it != pmap.end();) {
    if(it->first.second == owner) {
      ctx().persistent_bufs.erase(it->second);
      free(it->second);
      it = pmap.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace mkx
