#pragma once

// 呼び出し場所(__FILE__/__LINE__)+ownerポインタをキーに、Backend::allocしたバッファを使い回す汎用レジストリ。

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>

namespace mkx {

inline uint64_t persistent_location_hash(const char* file, int line) {
  return std::hash<std::string>{}(std::string(file) + ":" + std::to_string(line));
}

namespace detail {

using PersistentKey = std::pair<uint64_t, const void*>;

struct PersistentKeyHash {
  size_t operator()(const PersistentKey& k) const { return std::hash<uint64_t>{}(k.first) ^ (std::hash<const void*>{}(k.second) << 1); }
};

inline std::unordered_map<PersistentKey, void*, PersistentKeyHash>& persistent_registry() {
  static std::unordered_map<PersistentKey, void*, PersistentKeyHash> reg;
  return reg;
}

} // namespace detail

template <ComputeBackend Backend> void* persistent_buffer(const void* owner, uint64_t loc_id, size_t nbytes) {
  auto& reg = detail::persistent_registry();
  detail::PersistentKey key{loc_id, owner};
  auto it = reg.find(key);
  if(it != reg.end()) return it->second;
  auto* buf = Backend::alloc(nbytes);
  Backend::register_persistent(buf);
  reg.emplace(key, buf);
  return buf;
}

template <ComputeBackend Backend> void release_persistent_for_owner(const void* owner) {
  auto& reg = detail::persistent_registry();
  for(auto it = reg.begin(); it != reg.end();) {
    if(it->first.second == owner) {
      auto* buf = static_cast<typename Backend::Buffer*>(it->second);
      Backend::unregister_persistent(buf);
      Backend::free(buf);
      it = reg.erase(it);
    } else {
      ++it;
    }
  }
}

// evalする前に呼ぶこと(nodeの出力バッファをbufに固定し、Backend::allocをスキップさせる)。
inline void set_persistent_output(OpNode& node, void* buf) { node.preallocated_outputs = {buf}; }

} // namespace mkx

#define MKX_PERSISTENT_BUF(Backend, owner, nbytes) mkx::persistent_buffer<Backend>((owner), mkx::persistent_location_hash(__FILE__, __LINE__), (nbytes))
