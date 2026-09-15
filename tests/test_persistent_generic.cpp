#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/ops/elementwise.hpp>
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
