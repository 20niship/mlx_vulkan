#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/persistent.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> persistent_input(const void* owner, uint64_t loc_id, std::vector<float> data) {
  mkx::array<float, 1> a(mkx::Shape{static_cast<int64_t>(data.size())});
  size_t nbytes = data.size() * sizeof(float);
  auto* buf     = static_cast<VulkanBackend::Buffer*>(mkx::persistent_buffer<VulkanBackend>(owner, loc_id, nbytes));
  VulkanBackend::upload(buf, data.data(), nbytes);
  a.node()->gpu_buffer = buf;
  a.node()->evaluated  = true;
  return a;
}
} // namespace

TEST_CASE("persistent_buffer: 通常の二項演算ノードも複数iterationで同じBuffer*を再利用する") {
  int owner_token;
  const void* owner = &owner_token;

  void* buf1_first = nullptr;
  void* buf2_first = nullptr;
  void* buf3_first = nullptr;

  for(int iter = 0; iter < 8; iter++) {
    float base = static_cast<float>(iter) * 10.0f;
    auto a1    = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), {base + 1, base + 2, base + 3});
    auto a2    = persistent_input(owner, mkx::persistent_location_hash(__FILE__, __LINE__), {base + 10, base + 20, base + 30});

    auto a3       = mkx::add(a1, a2);
    size_t nbytes = 3 * sizeof(float);
    mkx::set_persistent_output(*a3.node(), mkx::persistent_buffer<VulkanBackend>(owner, mkx::persistent_location_hash(__FILE__, __LINE__), nbytes));
    mkx::eval<VulkanBackend>(a3);

    auto v = a3.to_vector<VulkanBackend>();
    CHECK(v[0] == doctest::Approx(2 * base + 11)); // a1={base+1,..}, a2={base+10,..}
    CHECK(v[1] == doctest::Approx(2 * base + 22));
    CHECK(v[2] == doctest::Approx(2 * base + 33));

    if(iter == 0) {
      buf1_first = a1.node()->gpu_buffer;
      buf2_first = a2.node()->gpu_buffer;
      buf3_first = a3.node()->gpu_buffer;
    } else {
      CHECK(a1.node()->gpu_buffer == buf1_first);
      CHECK(a2.node()->gpu_buffer == buf2_first);
      CHECK(a3.node()->gpu_buffer == buf3_first);
    }
  }

  mkx::release_persistent_for_owner<VulkanBackend>(owner);
}
