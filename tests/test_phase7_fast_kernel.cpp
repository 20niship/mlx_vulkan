#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/fast_kernel.hpp>
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

TEST_CASE("compute_kernel: 2入力2出力のカスタムカーネル基盤の疎通確認") {
  auto kernel = mkx::fast::compute_kernel("add_sub_kernel", {"a", "b"}, {"sum_out", "diff_out"},
                                          R"GLSL(
        uint i = gl_GlobalInvocationID.x;
        if (i >= 4u) return;
        sum_out[i] = a[i] + b[i];
        diff_out[i] = a[i] - b[i];
        )GLSL");

  auto a = make1({1, 2, 3, 4});
  auto b = make1({10, 20, 30, 40});

  auto outputs = kernel({a, b}, {mkx::Shape{4}, mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1});
  REQUIRE(outputs.size() == 2);

  mkx::eval<VulkanBackend>(outputs[0], outputs[1]);
  auto sum_v  = outputs[0].to_vector<VulkanBackend>();
  auto diff_v = outputs[1].to_vector<VulkanBackend>();

  std::vector<float> expected_sum  = {11, 22, 33, 44};
  std::vector<float> expected_diff = {-9, -18, -27, -36};
  for(int i = 0; i < 4; ++i) {
    CHECK(sum_v[i] == doctest::Approx(expected_sum[i]));
    CHECK(diff_v[i] == doctest::Approx(expected_diff[i]));
  }
}

TEST_CASE("compute_kernel: preallocated_outputsで渡した永続バッファを2回のdispatchで再利用できる") {
  auto kernel = mkx::fast::compute_kernel("scale_kernel", {"a"}, {"result"},
                                          R"GLSL(
        uint i = gl_GlobalInvocationID.x;
        if (i >= 4u) return;
        result[i] = a[i] * 2.0;
        )GLSL");

  auto* persistent = VulkanBackend::alloc(4 * sizeof(float));
  VulkanBackend::register_persistent(persistent);

  auto a1   = make1({1, 2, 3, 4});
  auto out1 = kernel({a1}, {mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1}, {persistent});
  mkx::eval<VulkanBackend>(out1[0]);
  auto v1 = out1[0].to_vector<VulkanBackend>();
  CHECK(v1[0] == doctest::Approx(2.0f));
  CHECK(v1[3] == doctest::Approx(8.0f));

  std::string stats_after_first = VulkanBackend::debug_stats();

  auto a2   = make1({5, 6, 7, 8});
  auto out2 = kernel({a2}, {mkx::Shape{4}}, {4, 1, 1}, {4, 1, 1}, {persistent});
  mkx::eval<VulkanBackend>(out2[0]);
  auto v2 = out2[0].to_vector<VulkanBackend>();
  CHECK(v2[0] == doctest::Approx(10.0f));
  CHECK(v2[3] == doctest::Approx(16.0f));

  std::string stats_after_second = VulkanBackend::debug_stats();
  CHECK(stats_after_first == stats_after_second); // 永続バッファはpoolに戻らずBackend::allocも発生しないため統計が不変

  VulkanBackend::unregister_persistent(persistent);
  VulkanBackend::free(persistent);
}
