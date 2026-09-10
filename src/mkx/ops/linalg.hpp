#pragma once

#include <mkx/core/array.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/shaders/shader_source.hpp>

namespace mkx {

template <class T, size_t N> array<T, 1> sum(const array<T, N>& a) { return array<T, 1>(make_node(OpType::Sum, Shape{1}, a.dtype(), {a.node()})); }

template <class T, size_t N> array<T, 1> reduce_max(const array<T, N>& a) { return array<T, 1>(make_node(OpType::ReduceMax, Shape{1}, a.dtype(), {a.node()})); }

template <class T, size_t N> array<T, 1> argmax(const array<T, N>& a) { return array<T, 1>(make_node(OpType::ArgMax, Shape{1}, a.dtype(), {a.node()})); }

template <class T, size_t N> array<T, 1> argmin(const array<T, N>& a) { return array<T, 1>(make_node(OpType::ArgMin, Shape{1}, a.dtype(), {a.node()})); }

template <class T, size_t N> array<T, N - 1> sum_axis(const array<T, N>& a, int axis) {
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape = a.shape();
  out_shape.erase(out_shape.begin() + axis);

  Push push;
  push.ndim = static_cast<uint32_t>(a.shape().size());
  for(size_t d = 0; d < a.shape().size(); ++d) {
    push.in_shape[d]   = static_cast<uint32_t>(a.shape()[d]);
    push.in_strides[d] = in_strides[d];
  }
  for(size_t d = 0; d < out_shape.size(); ++d) push.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  push.in_base_offset = static_cast<uint32_t>(axis);

  auto node = make_node(OpType::SumAxis, out_shape, a.dtype(), {a.node()}, pack_push(push));
  return array<T, N - 1>(node);
}

template <class T, size_t N> array<T, N - 1> reduce_max_axis(const array<T, N>& a, int axis) {
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape = a.shape();
  out_shape.erase(out_shape.begin() + axis);

  Push push;
  push.ndim = static_cast<uint32_t>(a.shape().size());
  for(size_t d = 0; d < a.shape().size(); ++d) {
    push.in_shape[d]   = static_cast<uint32_t>(a.shape()[d]);
    push.in_strides[d] = in_strides[d];
  }
  for(size_t d = 0; d < out_shape.size(); ++d) push.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  push.in_base_offset = static_cast<uint32_t>(axis);

  auto node = make_node(OpType::ReduceMaxAxis, out_shape, a.dtype(), {a.node()}, pack_push(push));
  return array<T, N - 1>(node);
}

template <class T> array<T, 2> matmul(const array<T, 2>& a, const array<T, 2>& b) {
  int64_t M  = a.shape()[0];
  int64_t K  = a.shape()[1];
  int64_t Nc = b.shape()[1];

  Push push;
  push.out_shape[0]   = static_cast<uint32_t>(M);
  push.out_shape[1]   = static_cast<uint32_t>(Nc);
  push.in_base_offset = static_cast<uint32_t>(K);

  auto node = make_node(OpType::MatMul, Shape{M, Nc}, a.dtype(), {a.node(), b.node()}, pack_push(push));
  return array<T, 2>(node);
}

template <class T> array<T, 2> cholesky(const array<T, 2>& a) { return array<T, 2>(make_node(OpType::Cholesky, a.shape(), a.dtype(), {a.node()})); }

template <class T> array<T, 1> solve_triangular(const array<T, 2>& l, const array<T, 1>& b) { return array<T, 1>(make_node(OpType::SolveTriangular, b.shape(), b.dtype(), {l.node(), b.node()})); }

template <class T, size_t N> array<T, N> cross(const array<T, N>& a, const array<T, N>& b) { return array<T, N>(make_node(OpType::Cross, a.shape(), a.dtype(), {a.node(), b.node()})); }

} // namespace mkx
