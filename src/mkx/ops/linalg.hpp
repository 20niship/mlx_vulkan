#pragma once

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/vmap_context.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/shaders/shader_source.hpp>

namespace mkx {

namespace detail {
// 公開reshape()/sum_axis()はvmap中に自動シフトするため、ここから呼ぶと二重補正になる。直接ノード構築で回避する。
template <class T, ComputeBackend Backend> array<T, 1, Backend> reduce_axis1_raw(OpType op_type, const NodePtr<Backend>& a_node, Dtype dt, const Shape& real_shape) {
  int64_t b = real_shape[0];
  Shape flat_shape{b, shape_size(real_shape) / b};
  auto flat_node = make_node<Backend>(OpType::Reshape, flat_shape, dt, {a_node});

  auto in_strides = row_major_strides(flat_shape);
  Push push;
  push.ndim           = 2;
  push.in_shape[0]    = static_cast<uint32_t>(flat_shape[0]);
  push.in_shape[1]    = static_cast<uint32_t>(flat_shape[1]);
  push.in_strides[0]  = in_strides[0];
  push.in_strides[1]  = in_strides[1];
  push.out_shape[0]   = static_cast<uint32_t>(b);
  push.in_base_offset = 1;

  auto node = make_node<Backend>(op_type, Shape{b}, dt, {flat_node}, pack_push(push));
  return array<T, 1, Backend>(node);
}
} // namespace detail

// vmap中はbatch軸(先頭)以外を全reduceする(非batch軸をflattenしSumAxisを呼ぶだけ、新規GLSL不要)。
template <class T, size_t N, ComputeBackend Backend> array<T, 1, Backend> sum(const array<T, N, Backend>& a) {
  if(in_vmap()) return detail::reduce_axis1_raw<T, Backend>(OpType::SumAxis, a.node(), a.dtype(), a.shape());
  return array<T, 1, Backend>(make_node<Backend>(OpType::Sum, Shape{1}, a.dtype(), {a.node()}));
}

template <class T, size_t N, ComputeBackend Backend> array<T, 1, Backend> reduce_max(const array<T, N, Backend>& a) {
  if(in_vmap()) return detail::reduce_axis1_raw<T, Backend>(OpType::ReduceMaxAxis, a.node(), a.dtype(), a.shape());
  return array<T, 1, Backend>(make_node<Backend>(OpType::ReduceMax, Shape{1}, a.dtype(), {a.node()}));
}

// ArgMax/ArgMinには軸指定カーネルが無いため、vmap中はCPUで行ごとに求める(Cholesky同様のCPU fallback)。
template <class T, size_t N, ComputeBackend Backend> array<T, 1, Backend> argmax(const array<T, N, Backend>& a) {
  if(in_vmap()) return array<T, 1, Backend>(make_node<Backend>(OpType::ArgMax, Shape{a.shape()[0]}, a.dtype(), {a.node()}));
  return array<T, 1, Backend>(make_node<Backend>(OpType::ArgMax, Shape{1}, a.dtype(), {a.node()}));
}

template <class T, size_t N, ComputeBackend Backend> array<T, 1, Backend> argmin(const array<T, N, Backend>& a) {
  if(in_vmap()) return array<T, 1, Backend>(make_node<Backend>(OpType::ArgMin, Shape{a.shape()[0]}, a.dtype(), {a.node()}));
  return array<T, 1, Backend>(make_node<Backend>(OpType::ArgMin, Shape{1}, a.dtype(), {a.node()}));
}

template <class T, size_t N, ComputeBackend Backend> array<T, N - 1, Backend> sum_axis(const array<T, N, Backend>& a, int axis) {
  int eff_axis    = in_vmap() ? axis + 1 : axis;
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape = a.shape();
  out_shape.erase(out_shape.begin() + eff_axis);

  Push push;
  push.ndim = static_cast<uint32_t>(a.shape().size());
  for(size_t d = 0; d < a.shape().size(); ++d) {
    push.in_shape[d]   = static_cast<uint32_t>(a.shape()[d]);
    push.in_strides[d] = in_strides[d];
  }
  for(size_t d = 0; d < out_shape.size(); ++d) push.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  push.in_base_offset = static_cast<uint32_t>(eff_axis);

  auto node = make_node<Backend>(OpType::SumAxis, out_shape, a.dtype(), {a.node()}, pack_push(push));
  return array<T, N - 1, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N - 1, Backend> reduce_max_axis(const array<T, N, Backend>& a, int axis) {
  int eff_axis    = in_vmap() ? axis + 1 : axis;
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape = a.shape();
  out_shape.erase(out_shape.begin() + eff_axis);

  Push push;
  push.ndim = static_cast<uint32_t>(a.shape().size());
  for(size_t d = 0; d < a.shape().size(); ++d) {
    push.in_shape[d]   = static_cast<uint32_t>(a.shape()[d]);
    push.in_strides[d] = in_strides[d];
  }
  for(size_t d = 0; d < out_shape.size(); ++d) push.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
  push.in_base_offset = static_cast<uint32_t>(eff_axis);

  auto node = make_node<Backend>(OpType::ReduceMaxAxis, out_shape, a.dtype(), {a.node()}, pack_push(push));
  return array<T, N - 1, Backend>(node);
}

// vmap中はaの実shapeが{B,M,K}(bも{B,K,N})になっている前提でバッチGEMMにする(matmul_glslがndim==3を見て分岐する)。
template <class T, ComputeBackend Backend> array<T, 2, Backend> matmul(const array<T, 2, Backend>& a, const array<T, 2, Backend>& b) {
  Push push;
  if(in_vmap()) {
    int64_t B           = a.shape()[0];
    int64_t M           = a.shape()[1];
    int64_t K           = a.shape()[2];
    int64_t Nc          = b.shape()[2];
    push.ndim           = 3;
    push.out_shape[0]   = static_cast<uint32_t>(B);
    push.out_shape[1]   = static_cast<uint32_t>(M);
    push.out_shape[2]   = static_cast<uint32_t>(Nc);
    push.in_base_offset = static_cast<uint32_t>(K);
    auto node           = make_node<Backend>(OpType::MatMul, Shape{B, M, Nc}, a.dtype(), {a.node(), b.node()}, pack_push(push));
    return array<T, 2, Backend>(node);
  }
  int64_t M           = a.shape()[0];
  int64_t K           = a.shape()[1];
  int64_t Nc          = b.shape()[1];
  push.out_shape[0]   = static_cast<uint32_t>(M);
  push.out_shape[1]   = static_cast<uint32_t>(Nc);
  push.in_base_offset = static_cast<uint32_t>(K);

  auto node = make_node<Backend>(OpType::MatMul, Shape{M, Nc}, a.dtype(), {a.node(), b.node()}, pack_push(push));
  return array<T, 2, Backend>(node);
}

template <class T, ComputeBackend Backend> array<T, 2, Backend> cholesky(const array<T, 2, Backend>& a) { return array<T, 2, Backend>(make_node<Backend>(OpType::Cholesky, a.shape(), a.dtype(), {a.node()})); }

template <class T, ComputeBackend Backend> array<T, 1, Backend> solve_triangular(const array<T, 2, Backend>& l, const array<T, 1, Backend>& b) { return array<T, 1, Backend>(make_node<Backend>(OpType::SolveTriangular, b.shape(), b.dtype(), {l.node(), b.node()})); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> cross(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return array<T, N, Backend>(make_node<Backend>(OpType::Cross, a.shape(), a.dtype(), {a.node(), b.node()})); }

} // namespace mkx
