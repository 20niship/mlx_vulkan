#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/forward.hpp"
#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

namespace {
mkx::array<float, 1> make1(std::vector<float> data) {
  return mkx::array<float, 1>(data, mkx::Shape{static_cast<int64_t>(data.size())});
}
} // namespace

TEST_CASE("forward kernel: 2body(world+hinge子)1関節1アクチュエータで実機dispatchが完走する") {
  const int nb = 2, nv = 1, nq = 1, nu = 1, njnt = 1;
  const float dt = 0.01f;

  auto xipos      = make1({0, 0, 0, 0, 0, 1});
  auto ximat      = make1({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1});
  auto xanchor    = make1({0, 0, 1});
  auto xaxis      = make1({0, 0, 1});
  auto xmat       = make1({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1});
  auto qpos       = make1({0.3f});
  auto qvel       = make1({0.5f});
  auto ctrl       = make1({0.2f});
  auto act_moment = make1({1.0f});

  auto kernel = mkx::mujoco::make_forward_kernel(nb, nv, nq, nu, njnt, dt, {0.0f, 0.0f, -9.81f},
                                                 /*body_parentid=*/{0, 0}, /*body_rootid=*/{0, 1},
                                                 /*body_mass=*/{0.0f, 1.0f}, /*body_inertia=*/{0, 0, 0, 0.1f, 0.1f, 0.1f},
                                                 /*dof_bodyid=*/{1}, /*dof_parentid=*/{-1},
                                                 /*dof_damping=*/{0.0f}, /*dof_armature=*/{0.0f}, /*dof_stiffness=*/{0.0f},
                                                 /*dof_qposadr=*/{0}, /*qpos_spring=*/{0.0f},
                                                 /*act_gain0=*/{1.0f}, /*act_bias0=*/{0.0f},
                                                 /*dof_jtype=*/{3}, /*dof_rotaxis=*/{-1}, /*dof_jid=*/{0},
                                                 /*body_dofs=*/{{}, {0}}, /*jnt_dofadr0=*/0);

  const int scratch_sz = nb * 10 + nv * 6 + nv * 6 + nb * 6 + nb * 6 + nb * 3 + nb + nv + nv;

  auto outputs = kernel({xipos, ximat, xanchor, xaxis, xmat, qpos, qvel, ctrl, act_moment}, {mkx::Shape{nv * nv}, mkx::Shape{nv}, mkx::Shape{nb * 3}, mkx::Shape{nb * 10}, mkx::Shape{nb * 6}, mkx::Shape{nv}, mkx::Shape{scratch_sz}, mkx::Shape{nv * 6}}, {1, 1, 1}, {1, 1, 1});
  REQUIRE(outputs.size() == 8);

  mkx::eval(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7]);

  auto qM            = outputs[0].to_vector();
  auto qfrc_smooth   = outputs[1].to_vector();
  auto subtree_com   = outputs[2].to_vector();
  auto cinert        = outputs[3].to_vector();
  auto cvel          = outputs[4].to_vector();
  auto qfrc_actuator = outputs[5].to_vector();
  auto cdof          = outputs[7].to_vector();

  REQUIRE(qM.size() == 1);
  REQUIRE(qfrc_smooth.size() == 1);
  REQUIRE(qfrc_actuator.size() == 1);
  REQUIRE(cdof.size() == 6);

  // 質量行列の対角成分は正定値であるはず(質量1.0の剛体+正則化)
  CHECK(qM[0] > 0.0f);
  // qfrc_actuator = gain*ctrl+bias = 1.0*0.2+0.0 = 0.2 (act_momentが単位なので直結)
  CHECK(qfrc_actuator[0] == doctest::Approx(0.2f));

  auto all_finite = [](const std::vector<float>& v) {
    for(float x : v) {
      if(!std::isfinite(x)) return false;
    }
    return true;
  };
  CHECK(all_finite(qM));
  CHECK(all_finite(qfrc_smooth));
  CHECK(all_finite(subtree_com));
  CHECK(all_finite(cinert));
  CHECK(all_finite(cvel));
  CHECK(all_finite(qfrc_actuator));
  CHECK(all_finite(cdof));
}
