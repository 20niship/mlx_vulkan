#pragma once

#include <cstdint>
#include <cstring>

namespace mkx {

// IEEE754 float32 <-> binary16 (round-to-nearest-even省略、単純truncationで十分な用途向け)
inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp   = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;

  if(exp <= 0) return static_cast<uint16_t>(sign); // アンダーフロー -> 0
  if(exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u); // オーバーフロー -> inf

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float half_to_float(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t x;

  if(exp == 0) {
    x = sign; // 非正規化数はゼロ扱いで十分
  } else if(exp == 0x1F) {
    x = sign | 0x7F800000u | (mant << 13); // inf/nan
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

} // namespace mkx
