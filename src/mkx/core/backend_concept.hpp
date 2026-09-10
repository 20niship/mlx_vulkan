#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

namespace mkx {

// eval<Backend>()で仮想関数を使わずコンパイル時にbackendを確定させるためのconcept。
template <class B>
concept ComputeBackend = requires(std::string_view src, size_t nbytes, std::span<typename B::Buffer*> bufs, std::span<const std::byte> push_data, std::array<uint32_t, 3> groups, size_t pipeline_hash) {
  typename B::Pipeline;
  typename B::Buffer;

  { B::compile(src, pipeline_hash) } -> std::same_as<typename B::Pipeline>;
  { B::alloc(nbytes) } -> std::same_as<typename B::Buffer*>;
  { B::free(std::declval<typename B::Buffer*>()) } -> std::same_as<void>;
  { B::upload(std::declval<typename B::Buffer*>(), std::declval<const void*>(), nbytes) } -> std::same_as<void>;
  { B::download(std::declval<typename B::Buffer*>(), std::declval<void*>(), nbytes) } -> std::same_as<void>;
  { B::dispatch(std::declval<typename B::Pipeline&>(), bufs, push_data, groups) } -> std::same_as<void>;
  { B::wait_idle() } -> std::same_as<void>;
};

} // namespace mkx
