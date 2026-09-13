#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> make1(std::vector<float> data) { return mkx::array<float, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())}); }
} // namespace

TEST_CASE("reshape/flatten: viewなのでgpu_bufferを共有する") {
  auto a = make1({1, 2, 3, 4, 5, 6});
  auto m = mkx::reshape(a, mkx::Shape{2, 3});
  mkx::eval(m);
  CHECK(mkx::buffer_for<VulkanBackend>(m.node()) == mkx::buffer_for<VulkanBackend>(a.node()));
  auto v = m.to_vector();
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(static_cast<float>(i + 1)));
}

TEST_CASE("transpose: 2x3行列を3x2に転置") {
  auto a = make1({1, 2, 3, 4, 5, 6});
  auto m = mkx::reshape(a, mkx::Shape{2, 3});
  auto t = mkx::transpose(m, {1, 0});
  mkx::eval(t);
  auto v = t.to_vector();
  // 元: [[1,2,3],[4,5,6]] -> 転置: [[1,4],[2,5],[3,6]]
  std::vector<float> expected = {1, 4, 2, 5, 3, 6};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("broadcast_to: 先頭次元を複製") {
  auto a = make1({1, 2, 3});
  auto b = mkx::broadcast_to(a, mkx::Shape{2, 3});
  mkx::eval(b);
  auto v                      = b.to_vector();
  std::vector<float> expected = {1, 2, 3, 1, 2, 3};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("tile: 2回繰り返す") {
  auto a = make1({1, 2, 3});
  auto t = mkx::tile(a, {2});
  mkx::eval(t);
  auto v                      = t.to_vector();
  std::vector<float> expected = {1, 2, 3, 1, 2, 3};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("slice: 部分区間を取り出す") {
  auto a = make1({10, 20, 30, 40, 50});
  auto s = mkx::slice(a, {1}, {4});
  mkx::eval(s);
  auto v = s.to_vector();
  REQUIRE(v.size() == 3);
  CHECK(v[0] == doctest::Approx(20.0f));
  CHECK(v[1] == doctest::Approx(30.0f));
  CHECK(v[2] == doctest::Approx(40.0f));
}

TEST_CASE("concatenate: axis0で連結") {
  auto a = make1({1, 2, 3});
  auto b = make1({4, 5});
  auto c = mkx::concatenate(a, b, 0);
  mkx::eval(c);
  auto v = c.to_vector();
  REQUIRE(v.size() == 5);
  std::vector<float> expected = {1, 2, 3, 4, 5};
  for(int i = 0; i < 5; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("stack: 新しい軸で積み重ねる") {
  auto a = make1({1, 2, 3});
  auto b = make1({4, 5, 6});
  auto s = mkx::stack<float, 2>(a, b, 0);
  mkx::eval(s);
  auto v = s.to_vector();
  REQUIRE(v.size() == 6);
  std::vector<float> expected = {1, 2, 3, 4, 5, 6};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("take: indexでgatherする") {
  auto data = make1({10, 20, 30, 40});
  auto idx  = make1({2, 0, 3});
  auto t    = mkx::take(data, idx, 0);
  mkx::eval(t);
  auto v = t.to_vector();
  REQUIRE(v.size() == 3);
  CHECK(v[0] == doctest::Approx(30.0f));
  CHECK(v[1] == doctest::Approx(10.0f));
  CHECK(v[2] == doctest::Approx(40.0f));
}

TEST_CASE("diag: ベクトルから対角行列") {
  auto v = make1({1, 2, 3});
  auto d = mkx::diag(v);
  mkx::eval(d);
  auto out = d.to_vector();
  REQUIRE(out.size() == 9);
  for(int r = 0; r < 3; ++r) {
    for(int c = 0; c < 3; ++c) {
      float expected = (r == c) ? static_cast<float>(r + 1) : 0.0f;
      CHECK(out[r * 3 + c] == doctest::Approx(expected));
    }
  }
}

TEST_CASE("tril/triu: 三角行列マスク") {
  auto v = make1({1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto m = mkx::reshape<float, 1, 2>(v, mkx::Shape{3, 3});

  auto lo = mkx::tril(m);
  mkx::eval(lo);
  auto lo_v                      = lo.to_vector();
  std::vector<float> expected_lo = {1, 0, 0, 4, 5, 0, 7, 8, 9};
  for(int i = 0; i < 9; ++i) CHECK(lo_v[i] == doctest::Approx(expected_lo[i]));

  auto up = mkx::triu(m);
  mkx::eval(up);
  auto up_v                      = up.to_vector();
  std::vector<float> expected_up = {1, 2, 3, 0, 5, 6, 0, 0, 9};
  for(int i = 0; i < 9; ++i) CHECK(up_v[i] == doctest::Approx(expected_up[i]));
}

TEST_CASE("copy: 値をそのまま複製する") {
  auto a = make1({7, 8, 9});
  auto c = mkx::copy(a);
  mkx::eval(c);
  auto v = c.to_vector();
  for(int i = 0; i < 3; ++i) CHECK(v[i] == doctest::Approx(static_cast<float>(7 + i)));
}
