#include <mkx/vulkan/vulkan_backend.hpp>

#include <mkx/shaders/shader_source.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace mkx {

namespace {

struct Context {
  VkInstance instance        = VK_NULL_HANDLE;
  VkPhysicalDevice physical  = VK_NULL_HANDLE;
  VkDevice device            = VK_NULL_HANDLE;
  VkQueue queue              = VK_NULL_HANDLE;
  uint32_t queue_family      = 0;
  VkCommandPool command_pool = VK_NULL_HANDLE;

  Context() {
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "mkx";
    app_info.apiVersion       = VK_API_VERSION_1_3;

    // MoltenVKはVK_KHR_portability_enumeration必須、Linux+lavapipe等には無いため存在する場合のみ有効化する。
    const char* portability_enum_ext = "VK_KHR_portability_enumeration";
    uint32_t inst_ext_count          = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, nullptr);
    std::vector<VkExtensionProperties> inst_exts(inst_ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &inst_ext_count, inst_exts.data());
    bool has_portability_enum = false;
    for(auto& e : inst_exts) {
      if(std::string_view(e.extensionName) == portability_enum_ext) has_portability_enum = true;
    }

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;
    if(has_portability_enum) {
      instance_info.flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
      instance_info.enabledExtensionCount   = 1;
      instance_info.ppEnabledExtensionNames = &portability_enum_ext;
    }
    if(vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("mkx: vkCreateInstance failed");
    }

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
  }
};

Context& ctx() {
  static Context context;
  return context;
}

// 旧実装はdispatch()ごとにcommand buffer alloc+submit+vkQueueWaitIdleしており、1 evalあたりの実dispatch数に比例したCPU側オーバーヘッドが支配的だった。同一eval()内のdispatchをここに積み、次のwait_idle()で1回のsubmit+waitにまとめる(descriptor poolは実行完了までpendingに保持)。
struct Batch {
  VkCommandBuffer cmd             = VK_NULL_HANDLE;
  bool open                       = false;
  bool has_prior_dispatch         = false;
  std::vector<VkDescriptorPool> pending_desc_pools;
  // 容量を使い切るまで使い回すpool(vkCreate/DestroyDescriptorPoolの呼び出し回数を減らす。尽きたら新しいpoolをpending_desc_poolsに追加)。
  VkDescriptorPool current_pool = VK_NULL_HANDLE;
  uint32_t current_pool_sets_used = 0;
  static constexpr uint32_t kPoolSetCapacity = 64;
  static constexpr uint32_t kPoolBindingCapacity = 4096;
};

Batch& batch() {
  static Batch b;
  return b;
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

std::unordered_map<size_t, std::vector<VulkanBackend::Buffer*>>& free_list() {
  static std::unordered_map<size_t, std::vector<VulkanBackend::Buffer*>> pool;
  return pool;
}

VulkanBackend::Buffer* VulkanBackend::alloc(size_t nbytes) {
  auto& pool = free_list();
  auto it    = pool.find(nbytes);
  if(it != pool.end() && !it->second.empty()) {
    auto* buf = it->second.back();
    it->second.pop_back();
    return buf;
  }

  auto& c   = ctx();
  auto* buf = new Buffer();
  buf->size = nbytes;

  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size        = nbytes;
  buffer_info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCreateBuffer(c.device, &buffer_info, nullptr, &buf->buffer);

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(c.device, buf->buffer, &reqs);
  VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.allocationSize  = reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(c.physical, reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(c.device, &alloc_info, nullptr, &buf->memory);
  vkBindBufferMemory(c.device, buf->buffer, buf->memory, 0);
  return buf;
}

void VulkanBackend::free(Buffer* buf) {
  if(!buf) return;
  auto& pool  = free_list();
  auto& slot  = pool[buf->size];
  if(slot.size() < kMaxPooledPerSize) {
    slot.push_back(buf);
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
  vkMapMemory(c.device, buf->memory, 0, nbytes, 0, &mapped);
  std::memcpy(mapped, data, nbytes);
  vkUnmapMemory(c.device, buf->memory);
}

void VulkanBackend::download(Buffer* buf, void* data, size_t nbytes) {
  // A dispatch may still be sitting unsubmitted in the open batch; flush so this read sees its output.
  flush_batch();
  auto& c      = ctx();
  void* mapped = nullptr;
  vkMapMemory(c.device, buf->memory, 0, nbytes, 0, &mapped);
  std::memcpy(data, mapped, nbytes);
  vkUnmapMemory(c.device, buf->memory);
}

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

  VkDescriptorSetAllocateInfo set_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  set_alloc.descriptorPool     = b.current_pool;
  set_alloc.descriptorSetCount = 1;
  set_alloc.pSetLayouts        = &pipeline.set_layout;
  VkDescriptorSet desc_set;
  if(vkAllocateDescriptorSets(c.device, &set_alloc, &desc_set) != VK_SUCCESS) {
    // Pool ran out of binding budget before its set-count budget (large pipelines); start a fresh pool.
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Batch::kPoolBindingCapacity};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets       = Batch::kPoolSetCapacity;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;
    vkCreateDescriptorPool(c.device, &pool_info, nullptr, &b.current_pool);
    b.pending_desc_pools.push_back(b.current_pool);
    b.current_pool_sets_used = 0;
    set_alloc.descriptorPool = b.current_pool;
    vkAllocateDescriptorSets(c.device, &set_alloc, &desc_set);
  }
  b.current_pool_sets_used++;

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
}

} // namespace mkx
