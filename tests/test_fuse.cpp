#include <doctest/doctest.h>

#include <cstdio>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/fuse.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
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

// broadcast_to(a_len1, {4}) -> add(., b) -> mul(., c) -> where(cond, mul_result, add_result)
mkx::array<float, 1> build_chain(mkx::array<float, 1> a_len1, mkx::array<float, 1> b, mkx::array<float, 1> c, mkx::array<float, 1> cond) {
  auto a_bc = mkx::broadcast_to(a_len1, mkx::Shape{4});
  auto t    = mkx::add(a_bc, b);
  auto u    = mkx::multiply(t, c);
  return mkx::where(cond, u, t);
}

size_t parse_pooled_count(const std::string& stats) {
  size_t n = 0;
  std::sscanf(stats.c_str(), "mkx pool: %zu", &n);
  return n;
}
} // namespace

TEST_CASE("eval_fused: broadcast+add+mul+whereの結果がeval<Backend>と一致する") {
  auto a_len1 = make1({10});
  auto b      = make1({1, 2, 3, 4});
  auto c      = make1({2, 2, 2, 2});
  auto cond   = make1({1, 0, 1, 0});

  auto unfused = build_chain(a_len1, b, c, cond);
  mkx::eval<VulkanBackend>(unfused);
  auto v_unfused = unfused.to_vector<VulkanBackend>();

  auto a_len1_2 = make1({10});
  auto b2       = make1({1, 2, 3, 4});
  auto c2       = make1({2, 2, 2, 2});
  auto cond2    = make1({1, 0, 1, 0});
  auto fused    = build_chain(a_len1_2, b2, c2, cond2);
  mkx::eval_fused<VulkanBackend>({fused.node()});
  auto v_fused = fused.to_vector<VulkanBackend>();

  for(int i = 0; i < 4; ++i) CHECK(v_fused[i] == doctest::Approx(v_unfused[i]));
}

TEST_CASE("eval_fused: fusion後は中間ノード分のバッファがpoolに増えない") {
  auto stats_before = VulkanBackend::debug_stats();
  size_t before      = parse_pooled_count(stats_before);

  auto a = make1({10});
  auto b = make1({1, 2, 3, 4});
  auto c = make1({2, 2, 2, 2});
  auto cond = make1({1, 0, 1, 0});
  auto fused = build_chain(a, b, c, cond);
  mkx::eval_fused<VulkanBackend>({fused.node()});
  fused.to_vector<VulkanBackend>();

  size_t after = parse_pooled_count(VulkanBackend::debug_stats());
  CHECK(after - before <= 2); // fusionでadd/mulはlocal変数化されbroadcast出力+where出力の2個分以下に収まるはず(非fusionなら4個)
}
