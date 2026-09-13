#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/collision.hpp"
#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("collision kernel: plane-meshで1つの接触点を検出する") {
  // geom0=plane(原点、法線+Z), geom1=mesh(1頂点、平面のz=0.3下に貫入)
  const int ng     = 2;
  const int npairs = 1;

  auto geom_xpos  = mkx::array<float, 1>::array1f({0, 0, 0, 0, 0, -0.3f}, mkx::Shape{});
  auto geom_xmat  = mkx::array<float, 1>::array1f({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1}, mkx::Shape{});
  auto mesh_verts = mkx::array<float, 1>::array1f({0, 0, 0}, mkx::Shape{});
  // pair_data: g1,g2,type1,type2,margin,rbound_sum
  auto pair_data    = mkx::array<float, 1>::array1f({0, 1, 0, 7, 0, 100}, mkx::Shape{});
  auto mesh_vertadr = mkx::array<float, 1>::array1f({0}, mkx::Shape{});
  auto mesh_vertnum = mkx::array<float, 1>::array1f({1}, mkx::Shape{});
  auto geom_dataid  = mkx::array<float, 1>::array1f({0, 0}, mkx::Shape{});

  auto kernel  = mkx::mujoco::make_collision_kernel(ng, npairs);
  auto outputs = kernel({geom_xpos, geom_xmat, mesh_verts, pair_data, mesh_vertadr, mesh_vertnum, geom_dataid}, {mkx::Shape{128 * 8}, mkx::Shape{1}}, {1, 1, 1}, {1, 1, 1});
  REQUIRE(outputs.size() == 2);

  mkx::eval(outputs[0], outputs[1]);
  auto contact_data  = outputs[0].to_vector();
  auto contact_count = outputs[1].to_vector();

  REQUIRE(contact_count.size() == 1);
  CHECK(contact_count[0] == doctest::Approx(1.0f));

  // contact_data[0..2]=pos, [3..5]=normal, [6]=dist, [7]=pair index
  CHECK(contact_data[0] == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(contact_data[1] == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(contact_data[2] == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(contact_data[3] == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(contact_data[4] == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(contact_data[5] == doctest::Approx(1.0f).epsilon(0.01));
  CHECK(contact_data[6] == doctest::Approx(-0.3f).epsilon(0.01));

  auto all_finite = [](const std::vector<float>& v) {
    for(float x : v) {
      if(!std::isfinite(x)) return false;
    }
    return true;
  };
  CHECK(all_finite(contact_data));
  CHECK(all_finite(contact_count));
}
