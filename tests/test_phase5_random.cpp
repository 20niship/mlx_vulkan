#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/random.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("random::normal: 同じseedなら同じ結果を再現する") {
  auto k = mkx::random::key(42);
  auto a = mkx::random::normal<float, 1>(k, {2000});
  auto b = mkx::random::normal<float, 1>(mkx::random::key(42), {2000});

  mkx::eval(a, b);
  auto va = a.to_vector();
  auto vb = b.to_vector();

  REQUIRE(va.size() == 2000);
  for(size_t i = 0; i < va.size(); ++i) CHECK(va[i] == doctest::Approx(vb[i]));
}

TEST_CASE("random::normal: 異なるseedなら異なる結果になる") {
  auto a = mkx::random::normal<float, 1>(mkx::random::key(1), {100});
  auto b = mkx::random::normal<float, 1>(mkx::random::key(2), {100});
  mkx::eval(a, b);
  auto va = a.to_vector();
  auto vb = b.to_vector();

  int diff_count = 0;
  for(size_t i = 0; i < va.size(); ++i) {
    if(std::abs(va[i] - vb[i]) > 1e-4f) diff_count++;
  }
  CHECK(diff_count > 90);
}

TEST_CASE("random::normal: 標準正規分布に近い平均・分散になる") {
  auto a = mkx::random::normal<float, 1>(mkx::random::key(7), {4000});
  mkx::eval(a);
  auto v = a.to_vector();

  double mean = 0.0;
  for(float x : v) mean += x;
  mean /= static_cast<double>(v.size());

  double var = 0.0;
  for(float x : v) var += (x - mean) * (x - mean);
  var /= static_cast<double>(v.size());

  CHECK(mean > -0.15);
  CHECK(mean < 0.15);
  CHECK(var > 0.8);
  CHECK(var < 1.2);
}
