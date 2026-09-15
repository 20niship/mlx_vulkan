#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("add/sub/mul/divはCPU計算と一致する") {
  auto a = mkx::array<float, 1>::array1f({1.0f, 2.0f, 3.0f}, mkx::Shape{});
  auto b = mkx::array<float, 1>::array1f({4.0f, 5.0f, 6.0f}, mkx::Shape{});

  auto add_r = mkx::add(a, b);
  mkx::eval(add_r);
  auto v = add_r.to_vector();
  CHECK(v[0] == doctest::Approx(5.0f));
  CHECK(v[1] == doctest::Approx(7.0f));
  CHECK(v[2] == doctest::Approx(9.0f));

  auto sub_r = mkx::subtract(b, a);
  mkx::eval(sub_r);
  v = sub_r.to_vector();
  CHECK(v[0] == doctest::Approx(3.0f));

  auto mul_r = mkx::multiply(a, b);
  mkx::eval(mul_r);
  v = mul_r.to_vector();
  CHECK(v[0] == doctest::Approx(4.0f));

  auto div_r = mkx::divide(b, a);
  mkx::eval(div_r);
  v = div_r.to_vector();
  CHECK(v[0] == doctest::Approx(4.0f));
}

TEST_CASE("sqrt/absなど単項演算") {
  auto a = mkx::array<float, 1>::array1f({4.0f, -9.0f, 16.0f}, mkx::Shape{});

  auto s = mkx::sqrt(a);
  mkx::eval(s);
  auto v = s.to_vector();
  CHECK(v[0] == doctest::Approx(2.0f));
  CHECK(v[2] == doctest::Approx(4.0f));

  auto ab = mkx::abs(a);
  mkx::eval(ab);
  v = ab.to_vector();
  CHECK(v[1] == doctest::Approx(9.0f));
}

TEST_CASE("比較・論理演算は0/1を返す") {
  auto a = mkx::array<float, 1>::array1f({1.0f, 2.0f, 3.0f}, mkx::Shape{});
  auto b = mkx::array<float, 1>::array1f({3.0f, 2.0f, 1.0f}, mkx::Shape{});

  auto gt = mkx::greater(a, b);
  mkx::eval(gt);
  auto v = gt.to_vector();
  CHECK(v[0] == doctest::Approx(0.0f));
  CHECK(v[1] == doctest::Approx(0.0f));
  CHECK(v[2] == doctest::Approx(1.0f));

  auto eq = mkx::equal(a, b);
  mkx::eval(eq);
  v = eq.to_vector();
  CHECK(v[1] == doctest::Approx(1.0f));
}

TEST_CASE("where/clip") {
  auto cond = mkx::array<float, 1>::array1f({1.0f, 0.0f, 1.0f}, mkx::Shape{});
  auto x    = mkx::array<float, 1>::array1f({10.0f, 20.0f, 30.0f}, mkx::Shape{});
  auto y    = mkx::array<float, 1>::array1f({-1.0f, -2.0f, -3.0f}, mkx::Shape{});

  auto w = mkx::where(cond, x, y);
  mkx::eval(w);
  auto v = w.to_vector();
  CHECK(v[0] == doctest::Approx(10.0f));
  CHECK(v[1] == doctest::Approx(-2.0f));
  CHECK(v[2] == doctest::Approx(30.0f));

  auto lo = mkx::array<float, 1>::array1f({0.0f, 0.0f, 0.0f}, mkx::Shape{});
  auto hi = mkx::array<float, 1>::array1f({5.0f, 5.0f, 5.0f}, mkx::Shape{});
  auto c  = mkx::clip(x, lo, hi);
  mkx::eval(c);
  v = c.to_vector();
  CHECK(v[0] == doctest::Approx(5.0f));
}
