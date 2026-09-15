#pragma once

#include <cstdint>
#include <vector>

namespace mkx {

enum class Dtype { Float32, Int32, Bool };

inline constexpr Dtype float32 = Dtype::Float32;
inline constexpr Dtype int32   = Dtype::Int32;
inline constexpr Dtype bool_   = Dtype::Bool;

using Shape = std::vector<int64_t>;

inline int64_t shape_size(const Shape& shape) {
  int64_t n = 1;
  for(auto d : shape) n *= d;
  return n;
}

inline size_t dtype_size(Dtype dt) {
  switch(dt) {
    case Dtype::Float32: return 4;
    case Dtype::Int32: return 4;
    case Dtype::Bool: return 4; // GPU側はuint32で保持
  }
  return 4;
}

} // namespace mkx
