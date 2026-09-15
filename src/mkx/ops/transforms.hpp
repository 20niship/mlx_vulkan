#pragma once

#include <utility>
#include <vector>

#include <functional>

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/vmap_context.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx {

namespace detail {

inline Shape drop_dim(Shape s, int axis) {
  s.erase(s.begin() + axis);
  return s;
}

inline Shape insert_dim(Shape s, int axis) {
  s.insert(s.begin() + axis, 1);
  return s;
}

} // namespace detail

// fnがarray<T,N,Backend>をそのまま受け取れるなら(shape非依存なelementwise系)fn(batched_in)を直接1回呼び、host loop無しの1 dispatchで済ませる。N-1専用fnはponytailのslice/concatenateループにフォールバック。
template <class T, size_t N, ComputeBackend Backend = VulkanBackend, class Fn> auto vmap(Fn fn, int in_axis, int out_axis) {
  return [fn, in_axis, out_axis](const array<T, N, Backend>& batched_in) -> array<T, N, Backend> {
    if constexpr(requires(const array<T, N, Backend>& x) { fn(x); }) {
      (void)in_axis;
      (void)out_axis;
      return fn(batched_in);
    } else {
      int64_t batch = batched_in.shape()[static_cast<size_t>(in_axis)];

      std::vector<int64_t> starts(batched_in.shape().size(), 0);
      std::vector<int64_t> stops = batched_in.shape();

      using OutUnbatched = decltype(fn(std::declval<array<T, N - 1, Backend>>()));
      std::vector<OutUnbatched> results;
      results.reserve(static_cast<size_t>(batch));

      for(int64_t b = 0; b < batch; ++b) {
        starts[static_cast<size_t>(in_axis)] = b;
        stops[static_cast<size_t>(in_axis)]  = b + 1;
        auto sliced                          = slice(batched_in, starts, stops);
        auto squeezed                        = reshape<T, N, N - 1, Backend>(sliced, detail::drop_dim(sliced.shape(), in_axis));
        results.push_back(fn(squeezed));
      }

      array<T, N, Backend> acc = reshape<T, N - 1, N, Backend>(results[0], detail::insert_dim(results[0].shape(), out_axis));
      for(size_t i = 1; i < results.size(); ++i) {
        auto r = reshape<T, N - 1, N, Backend>(results[i], detail::insert_dim(results[i].shape(), out_axis));
        acc    = concatenate(acc, r, out_axis);
      }
      return acc;
    }
  };
}

// mx.vmap(fn, in_axes=0, out_axes=0)準拠の複数入出力版。fnをグラフ複製なしで1回だけ呼ぶ(fn内部の演算はVmapGuard中自動でbatch軸を考慮)。
template <ComputeBackend Backend = VulkanBackend> std::vector<array<float, 1, Backend>> vmap(const std::function<std::vector<array<float, 1, Backend>>(const std::vector<array<float, 1, Backend>>&)>& fn, const std::vector<array<float, 1, Backend>>& batched_inputs, int in_axes = 0, int out_axes = 0) {
  (void)out_axes;
  int64_t batch_size = batched_inputs[0].shape()[static_cast<size_t>(in_axes)];
  VmapGuard guard(batch_size, in_axes);
  return fn(batched_inputs);
}

// fusionはeval()が常に行うため、compile()はpassthroughのままでよい。
template <class Fn> auto compile(Fn fn) { return fn; }

} // namespace mkx
