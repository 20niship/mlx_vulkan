#pragma once

#include <utility>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/ops/shape.hpp>

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

// fnがarray<T,N>をそのまま受け取れるなら(shape非依存なelementwise系)fn(batched_in)を直接1回呼び、host loop無しの1 dispatchで済ませる。N-1専用fnはponytailのslice/concatenateループにフォールバック。
template <class T, size_t N, class Fn> auto vmap(Fn fn, int in_axis, int out_axis) {
  return [fn, in_axis, out_axis](const array<T, N>& batched_in) -> array<T, N> {
    if constexpr(requires(const array<T, N>& x) { fn(x); }) {
      (void)in_axis;
      (void)out_axis;
      return fn(batched_in);
    } else {
      int64_t batch = batched_in.shape()[static_cast<size_t>(in_axis)];

      std::vector<int64_t> starts(batched_in.shape().size(), 0);
      std::vector<int64_t> stops = batched_in.shape();

      using OutUnbatched = decltype(fn(std::declval<array<T, N - 1>>()));
      std::vector<OutUnbatched> results;
      results.reserve(static_cast<size_t>(batch));

      for(int64_t b = 0; b < batch; ++b) {
        starts[static_cast<size_t>(in_axis)] = b;
        stops[static_cast<size_t>(in_axis)]  = b + 1;
        auto sliced                          = slice(batched_in, starts, stops);
        auto squeezed                        = reshape<T, N, N - 1>(sliced, detail::drop_dim(sliced.shape(), in_axis));
        results.push_back(fn(squeezed));
      }

      array<T, N> acc = reshape<T, N - 1, N>(results[0], detail::insert_dim(results[0].shape(), out_axis));
      for(size_t i = 1; i < results.size(); ++i) {
        auto r = reshape<T, N - 1, N>(results[i], detail::insert_dim(results[i].shape(), out_axis));
        acc    = concatenate(acc, r, out_axis);
      }
      return acc;
    }
  };
}

// ponytail: eval()がshaderソースのハッシュでpipelineを既にキャッシュしているためcompileは今のところpassthrough、複数ノードを1shaderに融合するfuse最適化は必要になったら追加。
template <class Fn> auto compile(Fn fn) { return fn; }

} // namespace mkx
