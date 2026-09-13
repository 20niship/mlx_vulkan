#include <doctest/doctest.h>

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

TEST_CASE("eval_fused: クラスタに閉じた中間ノードはgpu_bufferを持たない") {
  auto a    = make1({10});
  auto b    = make1({1, 2, 3, 4});
  auto c    = make1({2, 2, 2, 2});
  auto cond = make1({1, 0, 1, 0});

  auto a_bc   = mkx::broadcast_to(a, mkx::Shape{4});
  auto t      = mkx::add(a_bc, b);
  auto u      = mkx::multiply(t, c);
  auto result = mkx::where(cond, u, t);

  mkx::eval_fused<VulkanBackend>({result.node()});

  CHECK(a_bc.node()->gpu_buffer == nullptr); // add/mulに吸収されクラスタ内local変数のまま
  CHECK(t.node()->gpu_buffer != nullptr);    // whereから別クラスタとして読まれるため実体化
  CHECK(u.node()->gpu_buffer != nullptr);
  CHECK(result.node()->gpu_buffer != nullptr);
}
