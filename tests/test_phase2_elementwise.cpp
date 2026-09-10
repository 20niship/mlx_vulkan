#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> make(std::vector<float> data) {
  auto a = mkx::zeros<float, 1>({static_cast<int64_t>(data.size())});
  mkx::eval<VulkanBackend>(a);
  auto* buf = static_cast<VulkanBackend::Buffer*>(a.node()->gpu_buffer);
  VulkanBackend::upload(buf, data.data(), data.size() * sizeof(float));
  return a;
}
} // namespace

TEST_CASE("add/sub/mul/divはCPU計算と一致する") {
  auto a = make({1.0f, 2.0f, 3.0f});
  auto b = make({4.0f, 5.0f, 6.0f});

  auto add_r = mkx::add(a, b);
  mkx::eval<VulkanBackend>(add_r);
  auto v = add_r.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(5.0f));
  CHECK(v[1] == doctest::Approx(7.0f));
  CHECK(v[2] == doctest::Approx(9.0f));

  auto sub_r = mkx::subtract(b, a);
  mkx::eval<VulkanBackend>(sub_r);
  v = sub_r.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(3.0f));

  auto mul_r = mkx::multiply(a, b);
  mkx::eval<VulkanBackend>(mul_r);
  v = mul_r.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(4.0f));

  auto div_r = mkx::divide(b, a);
  mkx::eval<VulkanBackend>(div_r);
  v = div_r.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(4.0f));
}

TEST_CASE("sqrt/absなど単項演算") {
  auto a = make({4.0f, -9.0f, 16.0f});

  auto s = mkx::sqrt(a);
  mkx::eval<VulkanBackend>(s);
  auto v = s.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(2.0f));
  CHECK(v[2] == doctest::Approx(4.0f));

  auto ab = mkx::abs(a);
  mkx::eval<VulkanBackend>(ab);
  v = ab.to_vector<VulkanBackend>();
  CHECK(v[1] == doctest::Approx(9.0f));
}

TEST_CASE("比較・論理演算は0/1を返す") {
  auto a = make({1.0f, 2.0f, 3.0f});
  auto b = make({3.0f, 2.0f, 1.0f});

  auto gt = mkx::greater(a, b);
  mkx::eval<VulkanBackend>(gt);
  auto v = gt.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(0.0f));
  CHECK(v[1] == doctest::Approx(0.0f));
  CHECK(v[2] == doctest::Approx(1.0f));

  auto eq = mkx::equal(a, b);
  mkx::eval<VulkanBackend>(eq);
  v = eq.to_vector<VulkanBackend>();
  CHECK(v[1] == doctest::Approx(1.0f));
}

TEST_CASE("where/clip") {
  auto cond = make({1.0f, 0.0f, 1.0f});
  auto x    = make({10.0f, 20.0f, 30.0f});
  auto y    = make({-1.0f, -2.0f, -3.0f});

  auto w = mkx::where(cond, x, y);
  mkx::eval<VulkanBackend>(w);
  auto v = w.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(10.0f));
  CHECK(v[1] == doctest::Approx(-2.0f));
  CHECK(v[2] == doctest::Approx(30.0f));

  auto lo = make({0.0f, 0.0f, 0.0f});
  auto hi = make({5.0f, 5.0f, 5.0f});
  auto c  = mkx::clip(x, lo, hi);
  mkx::eval<VulkanBackend>(c);
  v = c.to_vector<VulkanBackend>();
  CHECK(v[0] == doctest::Approx(5.0f));
}
