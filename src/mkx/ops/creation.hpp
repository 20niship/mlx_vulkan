#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>

namespace mkx {

namespace detail {
inline std::vector<std::byte> pack2f(float p0, float p1) {
  std::vector<std::byte> out(sizeof(float) * 2);
  std::memcpy(out.data(), &p0, sizeof(float));
  std::memcpy(out.data() + sizeof(float), &p1, sizeof(float));
  return out;
}
} // namespace detail

template <class T, size_t N> array<T, N> zeros(Shape shape) {
  auto node = make_node(OpType::Fill, shape, Dtype::Float32, {}, detail::pack2f(0.0f, 0.0f));
  return array<T, N>(node);
}

template <class T, size_t N> array<T, N> ones(Shape shape) {
  auto node = make_node(OpType::Fill, shape, Dtype::Float32, {}, detail::pack2f(1.0f, 0.0f));
  return array<T, N>(node);
}

template <class T, size_t N> array<T, N> zeros_like(const array<T, N>& a) { return zeros<T, N>(a.shape()); }

template <class T> array<T, 2> eye(int64_t n) {
  auto node = make_node(OpType::Eye, Shape{n, n}, Dtype::Float32, {}, detail::pack2f(static_cast<float>(n), 0.0f));
  return array<T, 2>(node);
}

template <class T> array<T, 1> arange(float start, float stop, float step = 1.0f) {
  int64_t count = static_cast<int64_t>(std::ceil((stop - start) / step));
  if(count < 0) count = 0;
  auto node = make_node(OpType::Range, Shape{count}, Dtype::Float32, {}, detail::pack2f(start, step));
  return array<T, 1>(node);
}

template <class To, class From, size_t N> array<To, N> astype(const array<From, N>& a) {
  auto node = make_node(OpType::AsType, a.shape(), a.dtype(), {a.node()});
  return array<To, N>(node);
}

} // namespace mkx
