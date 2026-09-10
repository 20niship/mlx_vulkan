#include <doctest/doctest.h>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("zeros: 全要素が0") {
  auto a = mkx::zeros<float, 1>({5});
  mkx::eval<VulkanBackend>(a);
  for(auto x : a.to_vector<VulkanBackend>()) CHECK(x == doctest::Approx(0.0f));
}

TEST_CASE("ones: 全要素が1") {
  auto a = mkx::ones<float, 1>({5});
  mkx::eval<VulkanBackend>(a);
  for(auto x : a.to_vector<VulkanBackend>()) CHECK(x == doctest::Approx(1.0f));
}

TEST_CASE("eye: 単位行列") {
  auto a = mkx::eye<float>(3);
  mkx::eval<VulkanBackend>(a);
  auto v = a.to_vector<VulkanBackend>();
  REQUIRE(v.size() == 9);
  for(int r = 0; r < 3; ++r) {
    for(int c = 0; c < 3; ++c) {
      CHECK(v[r * 3 + c] == doctest::Approx(r == c ? 1.0f : 0.0f));
    }
  }
}

TEST_CASE("arange: 0から5未満を1刻み") {
  auto a = mkx::arange<float>(0.0f, 5.0f, 1.0f);
  mkx::eval<VulkanBackend>(a);
  auto v = a.to_vector<VulkanBackend>();
  REQUIRE(v.size() == 5);
  for(size_t i = 0; i < v.size(); ++i) CHECK(v[i] == doctest::Approx(static_cast<float>(i)));
}
