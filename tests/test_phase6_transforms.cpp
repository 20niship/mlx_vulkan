#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/ops/transforms.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> make1(std::vector<float> data) { return mkx::array<float, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())}); }
} // namespace

TEST_CASE("vmap: バッチ軸0で各環境ごとにsquareを適用する") {
  // batched shape (3 envs, 2 dof)
  auto flat    = make1({1, 2, 3, 4, 5, 6});
  auto batched = mkx::reshape<float, 1, 2>(flat, mkx::Shape{3, 2});

  auto square_fn = [](const mkx::array<float, 1>& x) { return mkx::square(x); };
  auto vsquare   = mkx::vmap<float, 2>(square_fn, 0, 0);

  auto out = vsquare(batched);
  mkx::eval(out);
  auto v = out.to_vector();
  REQUIRE(v.size() == 6);
  std::vector<float> expected = {1, 4, 9, 16, 25, 36};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("vmap: shape非依存な関数はfast pathで1 dispatch(host loop無し)になる") {
  auto flat    = make1({1, 2, 3, 4, 5, 6});
  auto batched = mkx::reshape<float, 1, 2>(flat, mkx::Shape{3, 2});

  // 任意のNを受け取れる汎用ラムダ(shape非依存)。vmapはfn(batched_in)を直接1回呼ぶfast pathを選ぶ。
  auto generic_square = []<class T, size_t N>(const mkx::array<T, N>& x) { return mkx::square(x); };
  auto vsquare        = mkx::vmap<float, 2>(generic_square, 0, 0);

  auto out = vsquare(batched);
  mkx::eval(out);
  auto v = out.to_vector();
  REQUIRE(v.size() == 6);
  std::vector<float> expected = {1, 4, 9, 16, 25, 36};
  for(int i = 0; i < 6; ++i) CHECK(v[i] == doctest::Approx(expected[i]));
}

TEST_CASE("compile: passthroughなので通常のeval結果と一致する") {
  auto a        = make1({1, 2, 3});
  auto b        = make1({4, 5, 6});
  auto add_fn   = [](const mkx::array<float, 1>& x, const mkx::array<float, 1>& y) { return mkx::add(x, y); };
  auto compiled = mkx::compile(add_fn);

  auto out = compiled(a, b);
  mkx::eval(out);
  auto v = out.to_vector();
  CHECK(v[0] == doctest::Approx(5.0f));
  CHECK(v[1] == doctest::Approx(7.0f));
  CHECK(v[2] == doctest::Approx(9.0f));
}
