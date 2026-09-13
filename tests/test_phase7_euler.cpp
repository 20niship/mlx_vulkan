#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/euler.hpp"
#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("euler kernel: nv=1の単純joint1本で半陰的Euler積分を1ステップ実行する") {
  // 1自由度(simple joint, qa=0, da=0), 質量4.0, 減衰なし
  float dt             = 0.01f;
  auto qM              = mkx::array<float, 1>::array1f({4.0f}, mkx::Shape{});
  auto qfrc_smooth     = mkx::array<float, 1>::array1f({1.0f}, mkx::Shape{});
  auto qfrc_constraint = mkx::array<float, 1>::array1f({0.0f}, mkx::Shape{});
  auto qvel_in         = mkx::array<float, 1>::array1f({0.5f}, mkx::Shape{});
  auto qpos_in         = mkx::array<float, 1>::array1f({0.2f}, mkx::Shape{});

  auto kernel = mkx::mujoco::make_euler_kernel(
    /*nv=*/1, /*nq=*/1, dt,
    /*simple_qa=*/{0}, /*simple_da=*/{0},
    /*free_joints=*/{}, /*ball_joints=*/{},
    /*dof_damping_vals=*/{0.0f});

  auto outputs = kernel({qM, qfrc_smooth, qfrc_constraint, qvel_in, qpos_in}, {mkx::Shape{1}, mkx::Shape{1}, mkx::Shape{1}}, {1, 1, 1}, {1, 1, 1});
  REQUIRE(outputs.size() == 3);

  mkx::eval(outputs[0], outputs[1], outputs[2]);
  auto qpos_out = outputs[0].to_vector();
  auto qvel_out = outputs[1].to_vector();
  auto qacc_out = outputs[2].to_vector();

  REQUIRE(qpos_out.size() == 1);
  REQUIRE(qvel_out.size() == 1);
  REQUIRE(qacc_out.size() == 1);

  // qacc = qfrc/qM (正則化1e-6は無視できるレベル)
  float expected_qacc = 1.0f / 4.0f;
  float expected_qvel = 0.5f + expected_qacc * dt;
  float expected_qpos = 0.2f + dt * expected_qvel;

  CHECK(qacc_out[0] == doctest::Approx(expected_qacc).epsilon(0.001));
  CHECK(qvel_out[0] == doctest::Approx(expected_qvel).epsilon(0.001));
  CHECK(qpos_out[0] == doctest::Approx(expected_qpos).epsilon(0.001));

  auto all_finite = [](const std::vector<float>& v) {
    for(float x : v) {
      if(!std::isfinite(x)) return false;
    }
    return true;
  };
  CHECK(all_finite(qpos_out));
  CHECK(all_finite(qvel_out));
  CHECK(all_finite(qacc_out));
}
