#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx {

namespace detail {
inline std::vector<std::byte> pack2f(float p0, float p1) {
  std::vector<std::byte> out(sizeof(float) * 2);
  std::memcpy(out.data(), &p0, sizeof(float));
  std::memcpy(out.data() + sizeof(float), &p1, sizeof(float));
  return out;
}
} // namespace detail

template <class T, size_t N, ComputeBackend Backend = VulkanBackend> array<T, N, Backend> zeros(Shape shape) {
  auto node = make_node<Backend>(OpType::Fill, shape, Dtype::Float32, {}, detail::pack2f(0.0f, 0.0f));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend = VulkanBackend> array<T, N, Backend> ones(Shape shape) {
  auto node = make_node<Backend>(OpType::Fill, shape, Dtype::Float32, {}, detail::pack2f(1.0f, 0.0f));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> zeros_like(const array<T, N, Backend>& a) { return zeros<T, N, Backend>(a.shape()); }

template <class T, ComputeBackend Backend = VulkanBackend> array<T, 2, Backend> eye(int64_t n) {
  auto node = make_node<Backend>(OpType::Eye, Shape{n, n}, Dtype::Float32, {}, detail::pack2f(static_cast<float>(n), 0.0f));
  return array<T, 2, Backend>(node);
}

template <class T, ComputeBackend Backend = VulkanBackend> array<T, 1, Backend> arange(float start, float stop, float step = 1.0f) {
  int64_t count = static_cast<int64_t>(std::ceil((stop - start) / step));
  if(count < 0) count = 0;
  auto node = make_node<Backend>(OpType::Range, Shape{count}, Dtype::Float32, {}, detail::pack2f(start, step));
  return array<T, 1, Backend>(node);
}

template <class To, class From, size_t N, ComputeBackend Backend> array<To, N, Backend> astype(const array<From, N, Backend>& a) {
  auto node = make_node<Backend>(OpType::AsType, a.shape(), a.dtype(), {a.node()});
  return array<To, N, Backend>(node);
}

} // namespace mkx
