#pragma once

#include <cstdint>

namespace mkx {

namespace detail {

struct VmapState {
  int64_t batch_size = 0;
  int axis           = 0;
  int depth          = 0;
};

inline VmapState& vmap_state() {
  static thread_local VmapState s;
  return s;
}

} // namespace detail

// vmap実行中はfn本体の演算(sum/sum_axis/matmul等)が自動でbatch軸を考慮するようにするRAIIガード。
struct VmapGuard {
  explicit VmapGuard(int64_t batch_size, int axis = 0) : prev_(detail::vmap_state()) { detail::vmap_state() = detail::VmapState{batch_size, axis, prev_.depth + 1}; }
  ~VmapGuard() { detail::vmap_state() = prev_; }
  VmapGuard(const VmapGuard&)            = delete;
  VmapGuard& operator=(const VmapGuard&) = delete;

private:
  detail::VmapState prev_;
};

inline bool in_vmap() { return detail::vmap_state().depth > 0; }
inline int vmap_axis() { return detail::vmap_state().axis; }
inline int64_t vmap_batch_size() { return detail::vmap_state().batch_size; }

} // namespace mkx
