#pragma once

#include <cstring>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/shaders/shader_source.hpp>

namespace mkx {

namespace detail {

inline std::vector<uint32_t> row_major_strides(const Shape& shape) {
  std::vector<uint32_t> strides(shape.size(), 1);
  for(int d = static_cast<int>(shape.size()) - 2; d >= 0; --d) {
    strides[d] = strides[d + 1] * static_cast<uint32_t>(shape[d + 1]);
  }
  return strides;
}

inline std::vector<std::byte> shape_push_bytes(Push push) { return pack_push(push); }

} // namespace detail

template <class T, size_t N, size_t M, ComputeBackend Backend> array<T, M, Backend> reshape(const array<T, N, Backend>& a, Shape new_shape) { return array<T, M, Backend>(make_node<Backend>(OpType::Reshape, std::move(new_shape), a.dtype(), {a.node()})); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> reshape(const array<T, N, Backend>& a, Shape new_shape) { return reshape<T, N, N, Backend>(a, std::move(new_shape)); }

template <class T, size_t N, ComputeBackend Backend> array<T, 1, Backend> flatten(const array<T, N, Backend>& a) { return array<T, 1, Backend>(make_node<Backend>(OpType::Flatten, Shape{shape_size(a.shape())}, a.dtype(), {a.node()})); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> transpose(const array<T, N, Backend>& a, std::vector<int> perm) {
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape(perm.size());
  Push push;
  push.ndim = static_cast<uint32_t>(perm.size());
  for(size_t d = 0; d < perm.size(); ++d) {
    out_shape[d]       = a.shape()[static_cast<size_t>(perm[d])];
    push.out_shape[d]  = static_cast<uint32_t>(out_shape[d]);
    push.in_shape[d]   = static_cast<uint32_t>(out_shape[d]);
    push.in_strides[d] = in_strides[static_cast<size_t>(perm[d])];
  }
  auto node = make_node<Backend>(OpType::Transpose, out_shape, a.dtype(), {a.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> broadcast_to(const array<T, N, Backend>& a, Shape target_shape) {
  Shape padded = a.shape();
  while(padded.size() < target_shape.size()) padded.insert(padded.begin(), 1);
  auto in_strides = detail::row_major_strides(padded);

  Push push;
  push.ndim = static_cast<uint32_t>(target_shape.size());
  for(size_t d = 0; d < target_shape.size(); ++d) {
    push.out_shape[d]  = static_cast<uint32_t>(target_shape[d]);
    push.in_shape[d]   = static_cast<uint32_t>(target_shape[d]);
    push.in_strides[d] = (padded[d] == 1 && target_shape[d] != 1) ? 0 : in_strides[d];
  }
  auto node = make_node<Backend>(OpType::BroadcastTo, target_shape, a.dtype(), {a.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> tile(const array<T, N, Backend>& a, std::vector<int64_t> reps) {
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape(a.shape().size());
  Push push;
  push.ndim = static_cast<uint32_t>(a.shape().size());
  for(size_t d = 0; d < a.shape().size(); ++d) {
    out_shape[d]       = a.shape()[d] * reps[d];
    push.out_shape[d]  = static_cast<uint32_t>(out_shape[d]);
    push.in_shape[d]   = static_cast<uint32_t>(a.shape()[d]);
    push.in_strides[d] = in_strides[d];
  }
  auto node = make_node<Backend>(OpType::Tile, out_shape, a.dtype(), {a.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> slice(const array<T, N, Backend>& a, std::vector<int64_t> starts, std::vector<int64_t> stops) {
  auto in_strides = detail::row_major_strides(a.shape());
  Shape out_shape(a.shape().size());
  Push push;
  push.ndim       = static_cast<uint32_t>(a.shape().size());
  uint32_t offset = 0;
  for(size_t d = 0; d < a.shape().size(); ++d) {
    out_shape[d]       = stops[d] - starts[d];
    push.out_shape[d]  = static_cast<uint32_t>(out_shape[d]);
    push.in_shape[d]   = static_cast<uint32_t>(out_shape[d]);
    push.in_strides[d] = in_strides[d];
    offset += static_cast<uint32_t>(starts[d]) * in_strides[d];
  }
  push.in_base_offset = offset;
  auto node           = make_node<Backend>(OpType::Slice, out_shape, a.dtype(), {a.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> concatenate(const array<T, N, Backend>& a, const array<T, N, Backend>& b, int axis) {
  auto a_strides                       = detail::row_major_strides(a.shape());
  auto b_strides                       = detail::row_major_strides(b.shape());
  Shape out_shape                      = a.shape();
  out_shape[static_cast<size_t>(axis)] = a.shape()[static_cast<size_t>(axis)] + b.shape()[static_cast<size_t>(axis)];

  Push push;
  push.ndim = static_cast<uint32_t>(out_shape.size());
  for(size_t d = 0; d < out_shape.size(); ++d) {
    push.out_shape[d]  = static_cast<uint32_t>(out_shape[d]);
    push.in_shape[d]   = a_strides[d]; // a_strides流用
    push.in_strides[d] = b_strides[d]; // b_strides流用
  }
  push.in_base_offset = static_cast<uint32_t>(axis);
  push.p0             = static_cast<float>(a.shape()[static_cast<size_t>(axis)]); // split

  auto node = make_node<Backend>(OpType::Concatenate, out_shape, a.dtype(), {a.node(), b.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> stack(const array<T, N - 1, Backend>& a, const array<T, N - 1, Backend>& b, int axis) {
  Shape unsq = a.shape();
  unsq.insert(unsq.begin() + axis, 1);
  auto a_view = reshape<T, N - 1, N, Backend>(a, unsq);
  auto b_view = reshape<T, N - 1, N, Backend>(b, unsq);
  return concatenate(a_view, b_view, axis);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> take(const array<T, N, Backend>& data, const array<float, 1, Backend>& indices, int axis) {
  auto data_strides                    = detail::row_major_strides(data.shape());
  Shape out_shape                      = data.shape();
  out_shape[static_cast<size_t>(axis)] = indices.shape()[0];

  Push push;
  push.ndim = static_cast<uint32_t>(out_shape.size());
  for(size_t d = 0; d < out_shape.size(); ++d) {
    push.out_shape[d]  = static_cast<uint32_t>(out_shape[d]);
    push.in_strides[d] = data_strides[d];
  }
  push.in_base_offset = static_cast<uint32_t>(axis);

  auto node = make_node<Backend>(OpType::Take, out_shape, data.dtype(), {data.node(), indices.node()}, detail::shape_push_bytes(push));
  return array<T, N, Backend>(node);
}

template <class T, ComputeBackend Backend> array<T, 2, Backend> diag(const array<T, 1, Backend>& v) {
  int64_t n = v.shape()[0];
  std::vector<std::byte> imm(sizeof(float) * 2);
  float p0 = static_cast<float>(n);
  float p1 = 0.0f;
  std::memcpy(imm.data(), &p0, sizeof(float));
  std::memcpy(imm.data() + sizeof(float), &p1, sizeof(float));
  auto node = make_node<Backend>(OpType::Diag, Shape{n, n}, v.dtype(), {v.node()}, imm);
  return array<T, 2, Backend>(node);
}

template <class T, ComputeBackend Backend> array<T, 2, Backend> tril(const array<T, 2, Backend>& a) {
  std::vector<std::byte> imm(sizeof(float) * 2);
  float p0 = static_cast<float>(a.shape()[1]);
  float p1 = 0.0f;
  std::memcpy(imm.data(), &p0, sizeof(float));
  std::memcpy(imm.data() + sizeof(float), &p1, sizeof(float));
  auto node = make_node<Backend>(OpType::Tril, a.shape(), a.dtype(), {a.node()}, imm);
  return array<T, 2, Backend>(node);
}

template <class T, ComputeBackend Backend> array<T, 2, Backend> triu(const array<T, 2, Backend>& a) {
  std::vector<std::byte> imm(sizeof(float) * 2);
  float p0 = static_cast<float>(a.shape()[1]);
  float p1 = 0.0f;
  std::memcpy(imm.data(), &p0, sizeof(float));
  std::memcpy(imm.data() + sizeof(float), &p1, sizeof(float));
  auto node = make_node<Backend>(OpType::Triu, a.shape(), a.dtype(), {a.node()}, imm);
  return array<T, 2, Backend>(node);
}

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> copy(const array<T, N, Backend>& a) { return array<T, N, Backend>(make_node<Backend>(OpType::Copy, a.shape(), a.dtype(), {a.node()})); }

} // namespace mkx
