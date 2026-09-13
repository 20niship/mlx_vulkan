#pragma once

#include <cstdint>
#include <cstring>

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx::random {

// mlx互換のkey API。実体はシード値のみを保持するホスト側の軽量構造体(GPU計算不要)。
struct Key {
  uint32_t seed;
};

inline Key key(uint32_t seed) { return Key{seed}; }

template <class T, size_t N, ComputeBackend Backend = VulkanBackend> array<T, N, Backend> normal(Key k, Shape shape) {
  float seed_as_float;
  std::memcpy(&seed_as_float, &k.seed, sizeof(float));
  std::vector<std::byte> imm(sizeof(float) * 2);
  std::memcpy(imm.data(), &seed_as_float, sizeof(float));
  float zero = 0.0f;
  std::memcpy(imm.data() + sizeof(float), &zero, sizeof(float));

  auto node = make_node<Backend>(OpType::RandomNormal, std::move(shape), Dtype::Float32, {}, imm);
  return array<T, N, Backend>(node);
}

} // namespace mkx::random
