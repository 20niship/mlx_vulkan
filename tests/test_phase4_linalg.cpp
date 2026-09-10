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

namespace {
mkx::array<float, 1> make1(std::vector<float> data) {
  auto a = mkx::zeros<float, 1>({static_cast<int64_t>(data.size())});
  mkx::eval<VulkanBackend>(a);
  auto* buf = static_cast<VulkanBackend::Buffer*>(a.node()->gpu_buffer);
  VulkanBackend::upload(buf, data.data(), data.size() * sizeof(float));
  return a;
}
} // namespace

TEST_CASE("sum: 全要素の総和") {
  auto a = make1({1, 2, 3, 4, 5});
  auto s = mkx::sum(a);
  mkx::eval<VulkanBackend>(s);
  auto v = s.to_vector<VulkanBackend>();
  REQUIRE(v.size() == 1);
  CHECK(v[0] == doctest::Approx(15.0f));
}

TEST_CASE("sum: 256要素を超えるgrid-strideリダクション") {
  std::vector<float> data(1000);
  for(size_t i = 0; i < data.size(); ++i) data[i] = 1.0f;
  auto a = make1(data);
  auto s = mkx::sum(a);
  mkx::eval<VulkanBackend>(s);
  auto v = s.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(1000.0f));
}

TEST_CASE("reduce_max/argmax/argmin") {
  auto a = make1({3, 1, 4, 1, 5, 9, 2, 6});

  auto mx = mkx::reduce_max(a);
  mkx::eval<VulkanBackend>(mx);
  CHECK(mx.to_vector<VulkanBackend>()[0] == doctest::Approx(9.0f));

  auto am = mkx::argmax(a);
  mkx::eval<VulkanBackend>(am);
  CHECK(am.to_vector<VulkanBackend>()[0] == doctest::Approx(5.0f));

  auto ami = mkx::argmin(a);
  mkx::eval<VulkanBackend>(ami);
  float idx = ami.to_vector<VulkanBackend>()[0];
  CHECK((idx == doctest::Approx(1.0f) || idx == doctest::Approx(3.0f)));
}

TEST_CASE("sum_axis/reduce_max_axis: 2Dの各軸方向リダクション") {
  auto flat = make1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto m    = mkx::reshape<float, 1, 2>(flat, mkx::Shape{3, 4});

  auto row_sums = mkx::sum_axis(m, 1);
  mkx::eval<VulkanBackend>(row_sums);
  auto rv = row_sums.to_vector<VulkanBackend>();
  REQUIRE(rv.size() == 3);
  CHECK(rv[0] == doctest::Approx(10.0f));
  CHECK(rv[1] == doctest::Approx(26.0f));
  CHECK(rv[2] == doctest::Approx(42.0f));

  auto col_sums = mkx::sum_axis(m, 0);
  mkx::eval<VulkanBackend>(col_sums);
  auto cv = col_sums.to_vector<VulkanBackend>();
  REQUIRE(cv.size() == 4);
  CHECK(cv[0] == doctest::Approx(15.0f));
  CHECK(cv[1] == doctest::Approx(18.0f));
  CHECK(cv[2] == doctest::Approx(21.0f));
  CHECK(cv[3] == doctest::Approx(24.0f));

  auto row_max = mkx::reduce_max_axis(m, 1);
  mkx::eval<VulkanBackend>(row_max);
  auto rm = row_max.to_vector<VulkanBackend>();
  CHECK(rm[0] == doctest::Approx(4.0f));
  CHECK(rm[1] == doctest::Approx(8.0f));
  CHECK(rm[2] == doctest::Approx(12.0f));
}

TEST_CASE("matmul: 2x3 * 3x2") {
  auto a1 = make1({1, 2, 3, 4, 5, 6});
  auto a  = mkx::reshape<float, 1, 2>(a1, mkx::Shape{2, 3});
  auto b1 = make1({7, 8, 9, 10, 11, 12});
  auto b  = mkx::reshape<float, 1, 2>(b1, mkx::Shape{3, 2});

  auto c = mkx::matmul(a, b);
  mkx::eval<VulkanBackend>(c);
  auto v = c.to_vector<VulkanBackend>();
  REQUIRE(v.size() == 4);
  // [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
  CHECK(v[0] == doctest::Approx(58.0f));
  CHECK(v[1] == doctest::Approx(64.0f));
  CHECK(v[2] == doctest::Approx(139.0f));
  CHECK(v[3] == doctest::Approx(154.0f));
}

TEST_CASE("cholesky + solve_triangular") {
  // A = [[4,2],[2,3]] (対称正定値) -> L=[[2,0],[1, sqrt(2)]]
  auto a1 = make1({4, 2, 2, 3});
  auto a  = mkx::reshape<float, 1, 2>(a1, mkx::Shape{2, 2});

  auto l = mkx::cholesky(a);
  mkx::eval<VulkanBackend>(l);
  auto lv = l.to_vector<VulkanBackend>();
  CHECK(lv[0] == doctest::Approx(2.0f));
  CHECK(lv[1] == doctest::Approx(0.0f));
  CHECK(lv[2] == doctest::Approx(1.0f));
  CHECK(lv[3] == doctest::Approx(std::sqrt(2.0f)));

  auto b = make1({4, 3});
  auto x = mkx::solve_triangular(l, b);
  mkx::eval<VulkanBackend>(x);
  auto xv = x.to_vector<VulkanBackend>();
  // L*x=b: 2*x0=4 -> x0=2; x0+sqrt(2)*x1=3 -> x1=(3-2)/sqrt(2)
  CHECK(xv[0] == doctest::Approx(2.0f));
  CHECK(xv[1] == doctest::Approx(1.0f / std::sqrt(2.0f)));
}

TEST_CASE("cross: 3要素ベクトルの外積") {
  auto a = make1({1, 0, 0});
  auto b = make1({0, 1, 0});
  auto c = mkx::cross(a, b);
  mkx::eval<VulkanBackend>(c);
  auto v = c.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(0.0f));
  CHECK(v[1] == doctest::Approx(0.0f));
  CHECK(v[2] == doctest::Approx(1.0f));
}
