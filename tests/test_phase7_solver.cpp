#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "mujoco/solver.hpp"
#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
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

int64_t scratch_size(int nv) {
  const int max_efc = 256;
  int S_H           = 0;
  int S_J           = S_H + nv * nv;
  int S_D           = S_J + max_efc * nv;
  int S_AREF        = S_D + max_efc;
  int S_FORCE       = S_AREF + max_efc;
  int S_GRAD        = S_FORCE + max_efc;
  int S_SEARCH      = S_GRAD + nv;
  int S_QACC        = S_SEARCH + nv;
  int S_MA          = S_QACC + nv;
  int S_JAREF       = S_MA + nv;
  int S_ACTIVE      = S_JAREF + max_efc;
  int S_MV          = S_ACTIVE + max_efc;
  int S_JV          = S_MV + nv;
  int S_JACP        = S_JV + max_efc;
  return S_JACP + nv * 3;
}
} // namespace

TEST_CASE("solver kernel: 接触無しなら早期returnでqfrc_constraint=0") {
  const int nb = 2, nv = 1;
  auto qM             = make1({1.0f});
  auto qfrc_smooth    = make1({0.0f});
  auto cdof           = make1({0, 0, 1, 0, 0, 0});
  auto subtree_com    = make1({0, 0, 0, 0, 0, 1});
  auto qvel           = make1({0.0f});
  auto contact_data   = make1(std::vector<float>(128 * 8, 0.0f));
  auto contact_count  = make1({0.0f});
  auto pair_props     = make1(std::vector<float>(18, 0.0f));
  auto body_dof_masks = make1({0.0f, 1.0f});
  auto body_rootid    = make1({0.0f, 1.0f});

  auto kernel  = mkx::mujoco::make_solver_kernel(nb, nv, 0.01f, /*use_pyramidal=*/false,
                                                 /*refsafe=*/true, /*impratio=*/1.0f,
                                                 /*solver_iters=*/1, /*cg_iters=*/15);
  auto outputs = kernel({qM, qfrc_smooth, cdof, subtree_com, qvel, contact_data, contact_count, pair_props, body_dof_masks, body_rootid}, {mkx::Shape{nv}, mkx::Shape{scratch_size(nv)}}, {static_cast<uint32_t>(nv), 1, 1}, {static_cast<uint32_t>(nv), 1, 1});
  REQUIRE(outputs.size() == 2);

  mkx::eval<VulkanBackend>(outputs[0]);
  auto qfrc_constraint = outputs[0].to_vector<VulkanBackend>();
  REQUIRE(qfrc_constraint.size() == 1);
  CHECK(qfrc_constraint[0] == doctest::Approx(0.0f));
}

TEST_CASE("solver kernel: 1接触(条件数1, 摩擦無し)でNewton+CG解が完走する") {
  const int nb = 2, nv = 1;
  auto qM          = make1({1.0f});
  auto qfrc_smooth = make1({0.0f});
  auto cdof        = make1({0, 0, 1, 0, 0, 0});
  auto subtree_com = make1({0, 0, 0, 0, 0, 0});
  auto qvel        = make1({0.0f});

  // 接触点(1,0,0)・法線(0,1,0): Z軸回転のdofが(1,0,0)を叩くとY方向速度になるのでヤコビアンが非0になる配置
  std::vector<float> cdata(128 * 8, 0.0f);
  cdata[0]           = 1;
  cdata[1]           = 0;
  cdata[2]           = 0; // pos
  cdata[3]           = 0;
  cdata[4]           = 1;
  cdata[5]           = 0;     // normal
  cdata[6]           = -0.1f; // dist(貫入)
  cdata[7]           = 0;     // pair index
  auto contact_data  = make1(cdata);
  auto contact_count = make1({1.0f});

  std::vector<float> pp(18, 0.0f);
  pp[0]               = 0;
  pp[1]               = 1;
  pp[2]               = 1;    // body1,body2,condim(摩擦無し)
  pp[3]               = 0.0f; // margin
  pp[9]               = 0.02f;
  pp[10]              = 1.0f; // solref
  pp[11]              = 0.9f;
  pp[12]              = 0.95f;
  pp[13]              = 0.001f;
  pp[14]              = 0.5f;
  pp[15]              = 2.0f; // solimp
  pp[16]              = 1.0f; // invweight_t
  auto pair_props     = make1(pp);
  auto body_dof_masks = make1({0.0f, 1.0f});
  auto body_rootid    = make1({0.0f, 1.0f});

  auto kernel  = mkx::mujoco::make_solver_kernel(nb, nv, 0.01f, /*use_pyramidal=*/false,
                                                 /*refsafe=*/true, /*impratio=*/1.0f,
                                                 /*solver_iters=*/1, /*cg_iters=*/15);
  auto outputs = kernel({qM, qfrc_smooth, cdof, subtree_com, qvel, contact_data, contact_count, pair_props, body_dof_masks, body_rootid}, {mkx::Shape{nv}, mkx::Shape{scratch_size(nv)}}, {static_cast<uint32_t>(nv), 1, 1}, {static_cast<uint32_t>(nv), 1, 1});

  mkx::eval<VulkanBackend>(outputs[0], outputs[1]);
  auto qfrc_constraint = outputs[0].to_vector<VulkanBackend>();
  auto scratch         = outputs[1].to_vector<VulkanBackend>();

  REQUIRE(qfrc_constraint.size() == 1);
  CHECK(std::isfinite(qfrc_constraint[0]));
  // Newton+CGの収束値の数値一致検証はPhase8待ち、ここでは完走とNaN/Inf無しのみ確認する。

  auto all_finite = [](const std::vector<float>& v) {
    for(float x : v) {
      if(!std::isfinite(x)) return false;
    }
    return true;
  };
  CHECK(all_finite(scratch));
}
