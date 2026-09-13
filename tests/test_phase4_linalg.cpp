#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("sum: 全要素の総和") {
  auto a = mkx::array<float, 1>::array1f({1, 2, 3, 4, 5}, mkx::Shape{});
  auto s = mkx::sum(a);
  mkx::eval(s);
  auto v = s.to_vector();
  REQUIRE(v.size() == 1);
  CHECK(v[0] == doctest::Approx(15.0f));
}

TEST_CASE("sum: 256要素を超えるgrid-strideリダクション") {
  std::vector<float> data(1000);
  for(size_t i = 0; i < data.size(); ++i) data[i] = 1.0f;
  auto a = mkx::array<float, 1>::array1f(data, mkx::Shape{});
  auto s = mkx::sum(a);
  mkx::eval(s);
  auto v = s.to_vector();
  CHECK(v[0] == doctest::Approx(1000.0f));
}

TEST_CASE("reduce_max/argmax/argmin") {
  auto a = mkx::array<float, 1>::array1f({3, 1, 4, 1, 5, 9, 2, 6}, mkx::Shape{});

  auto mx = mkx::reduce_max(a);
  mkx::eval(mx);
  CHECK(mx.to_vector()[0] == doctest::Approx(9.0f));

  auto am = mkx::argmax(a);
  mkx::eval(am);
  CHECK(am.to_vector()[0] == doctest::Approx(5.0f));

  auto ami = mkx::argmin(a);
  mkx::eval(ami);
  float idx = ami.to_vector()[0];
  CHECK((idx == doctest::Approx(1.0f) || idx == doctest::Approx(3.0f)));
}

TEST_CASE("sum_axis/reduce_max_axis: 2Dの各軸方向リダクション") {
  auto flat = mkx::array<float, 1>::array1f({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, mkx::Shape{});
  auto m    = mkx::reshape<float, 1, 2>(flat, mkx::Shape{3, 4});

  auto row_sums = mkx::sum_axis(m, 1);
  mkx::eval(row_sums);
  auto rv = row_sums.to_vector();
  REQUIRE(rv.size() == 3);
  CHECK(rv[0] == doctest::Approx(10.0f));
  CHECK(rv[1] == doctest::Approx(26.0f));
  CHECK(rv[2] == doctest::Approx(42.0f));

  auto col_sums = mkx::sum_axis(m, 0);
  mkx::eval(col_sums);
  auto cv = col_sums.to_vector();
  REQUIRE(cv.size() == 4);
  CHECK(cv[0] == doctest::Approx(15.0f));
  CHECK(cv[1] == doctest::Approx(18.0f));
  CHECK(cv[2] == doctest::Approx(21.0f));
  CHECK(cv[3] == doctest::Approx(24.0f));

  auto row_max = mkx::reduce_max_axis(m, 1);
  mkx::eval(row_max);
  auto rm = row_max.to_vector();
  CHECK(rm[0] == doctest::Approx(4.0f));
  CHECK(rm[1] == doctest::Approx(8.0f));
  CHECK(rm[2] == doctest::Approx(12.0f));
}

TEST_CASE("matmul: 2x3 * 3x2") {
  auto a1 = mkx::array<float, 1>::array1f({1, 2, 3, 4, 5, 6}, mkx::Shape{});
  auto a  = mkx::reshape<float, 1, 2>(a1, mkx::Shape{2, 3});
  auto b1 = mkx::array<float, 1>::array1f({7, 8, 9, 10, 11, 12}, mkx::Shape{});
  auto b  = mkx::reshape<float, 1, 2>(b1, mkx::Shape{3, 2});

  auto c = mkx::matmul(a, b);
  mkx::eval(c);
  auto v = c.to_vector();
  REQUIRE(v.size() == 4);
  // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
  CHECK(v[0] == doctest::Approx(58.0f));
  CHECK(v[1] == doctest::Approx(64.0f));
  CHECK(v[2] == doctest::Approx(139.0f));
  CHECK(v[3] == doctest::Approx(154.0f));
}

TEST_CASE("cholesky + solve_triangular") {
  // A = [[4,2],[2,3]] (対称正定値) -> L=[[2,0],[1, sqrt(2)]]
  auto a1 = mkx::array<float, 1>::array1f({4, 2, 2, 3}, mkx::Shape{});
  auto a  = mkx::reshape<float, 1, 2>(a1, mkx::Shape{2, 2});

  auto l = mkx::cholesky(a);
  mkx::eval(l);
  auto lv = l.to_vector();
  CHECK(lv[0] == doctest::Approx(2.0f));
  CHECK(lv[1] == doctest::Approx(0.0f));
  CHECK(lv[2] == doctest::Approx(1.0f));
  CHECK(lv[3] == doctest::Approx(std::sqrt(2.0f)));

  auto b = mkx::array<float, 1>::array1f({4, 3}, mkx::Shape{});
  auto x = mkx::solve_triangular(l, b);
  mkx::eval(x);
  auto xv = x.to_vector();
  // L*x=b: 2*x0=4 -> x0=2; x0+sqrt(2)*x1=3 -> x1=(3-2)/sqrt(2)
  CHECK(xv[0] == doctest::Approx(2.0f));
  CHECK(xv[1] == doctest::Approx(1.0f / std::sqrt(2.0f)));
}

TEST_CASE("cross: 3要素ベクトルの外積") {
  auto a = mkx::array<float, 1>::array1f({1, 0, 0}, mkx::Shape{});
  auto b = mkx::array<float, 1>::array1f({0, 1, 0}, mkx::Shape{});
  auto c = mkx::cross(a, b);
  mkx::eval(c);
  auto v = c.to_vector();
  CHECK(v[0] == doctest::Approx(0.0f));
  CHECK(v[1] == doctest::Approx(0.0f));
  CHECK(v[2] == doctest::Approx(1.0f));
}
