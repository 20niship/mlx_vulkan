#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/kinematics.hpp"
#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> make1(std::vector<float> data) { return mkx::array<float, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())}); }
} // namespace

TEST_CASE("kinematics kernel: 2body(world+hinge子)1関節1geomで実機dispatchが完走する") {
  // body0=world(root), body1=hinge軸(0,0,1)でbody0から(0,0,1)オフセット
  auto body_parentid = make1({0, 0});
  auto body_pos      = make1({0, 0, 0, 0, 0, 1});
  auto body_quat     = make1({1, 0, 0, 0, 1, 0, 0, 0});
  auto body_ipos     = make1({0, 0, 0, 0, 0, 0});
  auto body_iquat    = make1({1, 0, 0, 0, 1, 0, 0, 0});
  auto body_jntadr   = make1({-1, 0});
  auto body_jntnum   = make1({0, 1});
  auto jnt_type      = make1({3}); // HINGE
  auto jnt_qposadr   = make1({0});
  auto jnt_pos       = make1({0, 0, 0});
  auto jnt_axis      = make1({0, 0, 1});
  auto qpos0         = make1({0});
  auto geom_bodyid   = make1({1});
  auto geom_pos      = make1({0, 0, 0});
  auto geom_quat     = make1({1, 0, 0, 0});
  auto qpos          = make1({0.5f});

  auto kernel  = mkx::mujoco::make_kinematics_kernel(/*nbody=*/2, /*njnt=*/1, /*nq=*/1, /*ngeom=*/1);
  auto outputs = kernel({body_parentid, body_pos, body_quat, body_ipos, body_iquat, body_jntadr, body_jntnum, jnt_type, jnt_qposadr, jnt_pos, jnt_axis, qpos0, geom_bodyid, geom_pos, geom_quat, qpos},
                        {mkx::Shape{6}, mkx::Shape{8}, mkx::Shape{18}, mkx::Shape{6}, mkx::Shape{18}, mkx::Shape{3}, mkx::Shape{3}, mkx::Shape{3}, mkx::Shape{9}}, {1, 1, 1}, {1, 1, 1});

  REQUIRE(outputs.size() == 9);
  mkx::eval(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8]);

  auto xpos      = outputs[0].to_vector();
  auto xquat     = outputs[1].to_vector();
  auto xmat      = outputs[2].to_vector();
  auto xanchor   = outputs[5].to_vector();
  auto geom_xpos = outputs[7].to_vector();

  REQUIRE(xpos.size() == 6);
  REQUIRE(xquat.size() == 8);
  REQUIRE(xmat.size() == 18);
  REQUIRE(xanchor.size() == 3);
  REQUIRE(geom_xpos.size() == 3);

  // body0(root)はbody_pos[0..2]/body_quat[0..3]をそのまま素通しするはず
  CHECK(xpos[0] == doctest::Approx(0.0f));
  CHECK(xpos[1] == doctest::Approx(0.0f));
  CHECK(xpos[2] == doctest::Approx(0.0f));
  CHECK(xquat[0] == doctest::Approx(1.0f));
  CHECK(xquat[1] == doctest::Approx(0.0f));
  CHECK(xquat[2] == doctest::Approx(0.0f));
  CHECK(xquat[3] == doctest::Approx(0.0f));

  // NaN/Infが出ていないことを全出力で確認(数値的な物理正しさの照合はPhase8待ち)
  auto all_finite = [](const std::vector<float>& v) {
    for(float x : v) {
      if(!std::isfinite(x)) return false;
    }
    return true;
  };
  CHECK(all_finite(xpos));
  CHECK(all_finite(xquat));
  CHECK(all_finite(xmat));
  CHECK(all_finite(xanchor));
  CHECK(all_finite(geom_xpos));
}
