#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/euler_devmem.hpp"
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

TEST_CASE("euler_devmem kernel: nv=100の対角質量行列でL_scratch経由の積分が完走する") {
  const int nv = 100;
  const int nq = nv;
  float dt     = 0.01f;

  // 対角質量行列(dof iの質量は i+1)、他は0
  std::vector<float> qM(static_cast<size_t>(nv * nv), 0.0f);
  for(int i = 0; i < nv; i++) qM[static_cast<size_t>(i * nv + i)] = static_cast<float>(i + 1);
  auto qM_arr = make1(qM);

  std::vector<float> qfrc(static_cast<size_t>(nv), 1.0f);
  auto qfrc_smooth     = make1(qfrc);
  auto qfrc_constraint = make1(std::vector<float>(static_cast<size_t>(nv), 0.0f));
  auto qvel_in         = make1(std::vector<float>(static_cast<size_t>(nv), 0.0f));
  auto qpos_in         = make1(std::vector<float>(static_cast<size_t>(nq), 0.0f));

  std::vector<int> simple_qa, simple_da;
  for(int i = 0; i < nv; i++) {
    simple_qa.push_back(i);
    simple_da.push_back(i);
  }

  auto kernel = mkx::mujoco::make_euler_devmem_kernel(nv, nq, dt, simple_qa, simple_da, /*free_joints=*/{}, /*ball_joints=*/{},
                                                      /*dof_damping_vals=*/std::vector<float>(static_cast<size_t>(nv), 0.0f));

  auto outputs = kernel({qM_arr, qfrc_smooth, qfrc_constraint, qvel_in, qpos_in}, {mkx::Shape{nq}, mkx::Shape{nv}, mkx::Shape{nv}, mkx::Shape{static_cast<int64_t>(nv) * nv}}, {1, 1, 1}, {1, 1, 1});
  REQUIRE(outputs.size() == 4);

  mkx::eval(outputs[0], outputs[1], outputs[2], outputs[3]);
  auto qpos_out = outputs[0].to_vector();
  auto qvel_out = outputs[1].to_vector();
  auto qacc_out = outputs[2].to_vector();

  REQUIRE(qpos_out.size() == static_cast<size_t>(nq));
  REQUIRE(qvel_out.size() == static_cast<size_t>(nv));
  REQUIRE(qacc_out.size() == static_cast<size_t>(nv));

  // 対角質量行列なのでqacc[i] = qfrc[i] / (i+1) の解析解と一致するはず
  for(int i = 0; i < nv; i++) {
    float expected_qacc = 1.0f / static_cast<float>(i + 1);
    CHECK(qacc_out[static_cast<size_t>(i)] == doctest::Approx(expected_qacc).epsilon(0.01));
  }

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
