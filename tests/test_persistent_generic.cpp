#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> persistent_input(const void* owner, uint64_t loc_id, std::vector<float> data) {
  mkx::array<float, 1> a(data, mkx::Shape{static_cast<int64_t>(data.size())});
  mkx::mark_permanent<VulkanBackend>(a.node(), loc_id, owner);
  return a;
}
} // namespace

TEST_CASE("is_permanent: 通常の二項演算ノードも複数iterationで同じBufferを再利用する") {
  int owner_token;
  const void* owner = &owner_token;

  VulkanBackend::Buffer* buf1_first = nullptr;
  VulkanBackend::Buffer* buf2_first = nullptr;
  VulkanBackend::Buffer* buf3_first = nullptr;

  for(int iter = 0; iter < 8; iter++) {
    float base = static_cast<float>(iter) * 10.0f;
    auto a1    = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), {base + 1, base + 2, base + 3});
    auto a2    = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), {base + 10, base + 20, base + 30});

    auto a3 = mkx::add(a1, a2);
    mkx::mark_permanent<VulkanBackend>(a3.node(), mkx::persistent_location_hash(__FILE__, __LINE__), owner);
    mkx::eval(a3);

    auto v = a3.to_vector();
    CHECK(v[0] == doctest::Approx(2 * base + 11)); // a1={base+1,..}, a2={base+10,..}
    CHECK(v[1] == doctest::Approx(2 * base + 22));
    CHECK(v[2] == doctest::Approx(2 * base + 33));

    auto* buf1 = mkx::buffer_for<VulkanBackend>(a1.node());
    auto* buf2 = mkx::buffer_for<VulkanBackend>(a2.node());
    auto* buf3 = mkx::buffer_for<VulkanBackend>(a3.node());
    if(iter == 0) {
      buf1_first = buf1;
      buf2_first = buf2;
      buf3_first = buf3;
    } else {
      CHECK(buf1 == buf1_first);
      CHECK(buf2 == buf2_first);
      CHECK(buf3 == buf3_first);
    }
  }

  VulkanBackend::release_persistent_for_owner(owner);
}

TEST_CASE("eval_cached: 2回目以降はbuilderを呼ばずreplayのみで最新のバッファ内容を反映する") {
  int owner_token;
  const void* owner = &owner_token;
  const int n       = 8;

  auto a = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), std::vector<float>(n, 1.0f));
  auto b = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), std::vector<float>(n, 2.0f));

  int build_calls = 0;
  auto build      = [&]() -> std::vector<mkx::NodePtr<VulkanBackend>> {
    build_calls++;
    auto s = mkx::sum(mkx::multiply(a, b));
    mkx::mark_permanent<VulkanBackend>(s.node(), mkx::persistent_location_hash(__FILE__, __LINE__), owner);
    return {s.node()};
  };

  auto roots1 = mkx::eval_cached<VulkanBackend>(owner, build);
  mkx::array<float, 1> s1(roots1[0]);
  CHECK(s1.to_vector()[0] == doctest::Approx(16.0f)); // 1*2*8
  CHECK(build_calls == 1);
  CHECK(VulkanBackend::has_replay(owner));

  std::vector<float> a_new(n, 3.0f);
  VulkanBackend::upload(VulkanBackend::get_or_allocate(a.node().get(), n * sizeof(float)), a_new.data(), n * sizeof(float));
  auto roots2 = mkx::eval_cached<VulkanBackend>(owner, build);
  mkx::array<float, 1> s2(roots2[0]);
  CHECK(s2.to_vector()[0] == doctest::Approx(48.0f)); // 3*2*8
  CHECK(build_calls == 1);                            // builderは呼ばれていない(replayのみ)

  VulkanBackend::invalidate_replay(owner);
  VulkanBackend::release_persistent_for_owner(owner);
}
