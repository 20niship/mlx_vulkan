#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("グラフ構築: addは即計算せずノードを積むだけ") {
  mkx::array<float, 1> a({4});
  mkx::array<float, 1> b({4});
  auto c = mkx::add(a, b);

  CHECK(c.node()->type == mkx::OpType::Add);
  CHECK_FALSE(c.node()->evaluated);
  CHECK(c.node()->inputs.size() == 2);
}

TEST_CASE("eval: Vulkan実機でaddを実行しCPUへ読み出せる") {
  auto a = mkx::zeros<float, 1>({4});
  auto b = mkx::ones<float, 1>({4});
  auto c = mkx::add(a, b);

  mkx::eval<VulkanBackend>(c);
  CHECK(c.node()->evaluated);

  auto v = c.to_vector<VulkanBackend>();
  REQUIRE(v.size() == 4);
  for(auto x : v) CHECK(x == doctest::Approx(1.0f));
}
