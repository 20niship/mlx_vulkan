#pragma once

#include <cassert>
#include <cstring>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/half.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx {

template <class T, size_t N, ComputeBackend Backend = VulkanBackend> class array {
public:
  explicit array(Shape shape, Dtype dtype = dtype_of()) : node_(make_node<Backend>(OpType::Const, std::move(shape), dtype)) {}

  explicit array(NodePtr<Backend> node) : node_(std::move(node)) {}

  array(std::vector<T> data, Shape shape, Dtype dtype = dtype_of()) : node_(make_node<Backend>(OpType::Const, std::move(shape), dtype)) {
    if constexpr(std::is_same_v<T, float>) {
      if(dtype == Dtype::Float16) {
        node_->host_data.resize(data.size() * sizeof(uint16_t));
        auto* dst = reinterpret_cast<uint16_t*>(node_->host_data.data());
        for(size_t i = 0; i < data.size(); i++) dst[i] = float_to_half(data[i]);
        return;
      }
    }
    node_->host_data.resize(data.size() * sizeof(T));
    std::memcpy(node_->host_data.data(), data.data(), node_->host_data.size());
  }


  template <ComputeBackend B = Backend> static array array1f(const std::vector<float>& data, Shape shape) { //
    return mkx::array<float, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())});
  }
  template <ComputeBackend B = Backend> static array array1i(const std::vector<int32_t>& data, Shape shape) { //
    return mkx::array<int32_t, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())});
  }
  template <ComputeBackend B = Backend> static array array2f(const std::vector<float>& data, Shape shape) { //
    return mkx::array<float, 2>(data, mkx::Shape{static_cast<int64_t>(data.size() / shape[1]), shape[1]});
  }

  const Shape& shape() const { return node_->shape; }
  Dtype dtype() const { return node_->dtype; }

  NodePtr<Backend>& node() { return node_; }
  const NodePtr<Backend>& node() const { return node_; }

  std::vector<T> to_vector() const {
    assert(node_->evaluated && "eval(arr) must be called before to_vector()");
    auto* buf = buffer_for<Backend>(node_);
    size_t n  = static_cast<size_t>(shape_size(node_->shape));
    std::vector<T> out(n);
    if constexpr(std::is_same_v<T, float>) {
      if(node_->dtype == Dtype::Float16) {
        std::vector<uint16_t> raw(n);
        Backend::download(buf, raw.data(), raw.size() * sizeof(uint16_t));
        for(size_t i = 0; i < n; i++) out[i] = half_to_float(raw[i]);
        return out;
      }
    }
    Backend::download(buf, out.data(), out.size() * sizeof(T));
    return out;
  }

  static constexpr Dtype dtype_of() {
    if constexpr(std::is_same_v<T, float>)
      return Dtype::Float32;
    else if constexpr(std::is_same_v<T, int32_t>)
      return Dtype::Int32;
    else
      return Dtype::Bool;
  }

private:
  NodePtr<Backend> node_;
};

} // namespace mkx
