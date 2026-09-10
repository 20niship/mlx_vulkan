#pragma once

#include <cassert>
#include <cstring>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/core/types.hpp>

namespace mkx {

template <class T, size_t N> class array {
public:
  array(Shape shape, DeviceTag device = DeviceTag::Vulkan) : device_(device), node_(make_node(OpType::Const, std::move(shape), dtype_of<T>())) {}

  explicit array(NodePtr node, DeviceTag device = DeviceTag::Vulkan) : device_(device), node_(std::move(node)) {}

  const Shape& shape() const { return node_->shape; }
  Dtype dtype() const { return node_->dtype; }
  DeviceTag device() const { return device_; }

  NodePtr& node() { return node_; }
  const NodePtr& node() const { return node_; }

  template <ComputeBackend Backend> std::vector<T> to_vector() const {
    assert(node_->evaluated && "eval<Backend>(arr) must be called before to_vector()");
    auto* buf = static_cast<typename Backend::Buffer*>(node_->gpu_buffer);
    std::vector<T> out(static_cast<size_t>(shape_size(node_->shape)));
    Backend::download(buf, out.data(), out.size() * sizeof(T));
    return out;
  }

private:
  template <class U> static constexpr Dtype dtype_of() {
    if constexpr(std::is_same_v<U, float>)
      return Dtype::Float32;
    else if constexpr(std::is_same_v<U, int32_t>)
      return Dtype::Int32;
    else
      return Dtype::Bool;
  }

  DeviceTag device_;
  NodePtr node_;
};

} // namespace mkx
