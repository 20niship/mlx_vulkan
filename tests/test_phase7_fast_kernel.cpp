#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/fast_kernel.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("compute_kernel: 2入力2出力のカスタムカーネル基盤の疎通確認") {
  auto kernel = mkx::fast::compute_kernel("add_sub_kernel", {"a", "b"}, {"sum_out", "diff_out"},
                                          R"GLSL(
        uint i = gl_GlobalInvocationID.x;
        if (i >= 4u) return;
        sum_out[i] = a[i] + b[i];
        diff_out[i] = a[i] - b[i];
        )GLSL");

  auto a = mkx::array<float, 1>::array1f({1, 2, 3, 4}, mkx::Shape{});
  auto b = mkx::array<float, 1>::array1f({10, 20, 30, 40}, mkx::Shape{});

  auto outputs = kernel({a, b}, {mkx::Shape{4}, mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1});
  REQUIRE(outputs.size() == 2);

  mkx::eval(outputs[0], outputs[1]);
  auto sum_v  = outputs[0].to_vector();
  auto diff_v = outputs[1].to_vector();

  std::vector<float> expected_sum  = {11, 22, 33, 44};
  std::vector<float> expected_diff = {-9, -18, -27, -36};
  for(int i = 0; i < 4; ++i) {
    CHECK(sum_v[i] == doctest::Approx(expected_sum[i]));
    CHECK(diff_v[i] == doctest::Approx(expected_diff[i]));
  }
}

TEST_CASE("compute_kernel: is_permanentな出力ノードは2回のdispatchで同じBufferを再利用する") {
  auto kernel = mkx::fast::compute_kernel("scale_kernel", {"a"}, {"result"},
                                          R"GLSL(
        uint i = gl_GlobalInvocationID.x;
        if (i >= 4u) return;
        result[i] = a[i] * 2.0;
        )GLSL");

  int owner_token;
  const void* owner = &owner_token;
  uint64_t loc_id   = mkx::persistent_location_hash(__FILE__, __LINE__);

  auto a1   = mkx::array<float, 1>::array1f({1, 2, 3, 4}, mkx::Shape{});
  auto out1 = kernel({a1}, {mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1});
  mkx::mark_permanent<VulkanBackend>(out1[0].node(), loc_id, owner);
  mkx::eval(out1[0]);
  auto v1 = out1[0].to_vector();
  CHECK(v1[0] == doctest::Approx(2.0f));
  CHECK(v1[3] == doctest::Approx(8.0f));
  auto* buf1 = mkx::buffer_for<VulkanBackend>(out1[0].node());

  auto a2   = mkx::array<float, 1>::array1f({5, 6, 7, 8}, mkx::Shape{});
  auto out2 = kernel({a2}, {mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1});
  mkx::mark_permanent<VulkanBackend>(out2[0].node(), loc_id, owner);
  mkx::eval(out2[0]);
  auto v2 = out2[0].to_vector();
  CHECK(v2[0] == doctest::Approx(10.0f));
  CHECK(v2[3] == doctest::Approx(16.0f));
  auto* buf2 = mkx::buffer_for<VulkanBackend>(out2[0].node());

  CHECK(buf1 == buf2);

  VulkanBackend::release_persistent_for_owner(owner);
}
