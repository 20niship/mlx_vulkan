#pragma once

#include <cassert>
#include <cstring>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx {

template <class T, size_t N, ComputeBackend Backend = VulkanBackend> class array {
public:
  explicit array(Shape shape) : node_(make_node<Backend>(OpType::Const, std::move(shape), dtype_of())) {}

  explicit array(NodePtr<Backend> node) : node_(std::move(node)) {}

  array(std::vector<T> data, Shape shape) : node_(make_node<Backend>(OpType::Const, std::move(shape), dtype_of())) {
    node_->host_data.resize(data.size() * sizeof(T));
    std::memcpy(node_->host_data.data(), data.data(), node_->host_data.size());
  }


  template <ComputeBackend B = Backend>
  explicit array1f(const std::vector<float>& data, Shape shape)
    requires std::is_same_v<B, VulkanBackend>

  const Shape& shape() const {
    return node_->shape;
  }
  Dtype dtype() const { return node_->dtype; }

  NodePtr<Backend>& node() { return node_; }
  const NodePtr<Backend>& node() const { return node_; }

  std::vector<T> to_vector() const {
    assert(node_->evaluated && "eval(arr) must be called before to_vector()");
    auto* buf = buffer_for<Backend>(node_);
    std::vector<T> out(static_cast<size_t>(shape_size(node_->shape)));
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
