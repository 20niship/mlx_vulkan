#include <doctest/doctest.h>

#include <chrono>
#include <random>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/ops/shape.hpp>
#include <mkx/ops/transforms.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;
using Arr = mkx::array<float, 1, VulkanBackend>;

namespace {

// 単一env用の合成関数(add/multiply/sum(軸なしreduce)/reshape/matmulを組み合わせ)。vmap有無どちらでも同じコードで動く。
std::vector<Arr> compute_fn(const std::vector<Arr>& in, int64_t k) {
  auto a      = in[0];
  auto b      = in[1];
  auto m_flat = in[2];

  auto c = mkx::add(a, b);
  auto s = mkx::sum(mkx::multiply(a, b));

  auto m  = mkx::reshape<float, 1, 2, VulkanBackend>(m_flat, mkx::Shape{k, k});
  auto mm = mkx::flatten(mkx::matmul(m, m));

  return {c, s, mm};
}

// 参照実装: envごとにslice->compute_fn->concatenateする素朴なループ(グラフはenv数だけ複製される)。
std::vector<Arr> reference_loop(const std::vector<Arr>& batched, int64_t batch, int64_t n, int64_t k) {
  std::vector<Arr> c_pieces, s_pieces, mm_pieces;
  for(int64_t e = 0; e < batch; ++e) {
    Arr a    = mkx::reshape<float, 1, 1, VulkanBackend>(mkx::slice(batched[0], {e * n}, {(e + 1) * n}), mkx::Shape{n});
    Arr b    = mkx::reshape<float, 1, 1, VulkanBackend>(mkx::slice(batched[1], {e * n}, {(e + 1) * n}), mkx::Shape{n});
    Arr m    = mkx::reshape<float, 1, 1, VulkanBackend>(mkx::slice(batched[2], {e * k * k}, {(e + 1) * k * k}), mkx::Shape{k * k});
    auto out = compute_fn({a, b, m}, k);
    c_pieces.push_back(out[0]);
    s_pieces.push_back(out[1]);
    mm_pieces.push_back(out[2]);
  }
  auto concat_all = [](std::vector<Arr>& pieces) {
    Arr acc = pieces[0];
    for(size_t i = 1; i < pieces.size(); ++i) acc = mkx::concatenate(acc, pieces[i], 0);
    return acc;
  };
  return {concat_all(c_pieces), concat_all(s_pieces), concat_all(mm_pieces)};
}

Arr make_flat(const std::vector<float>& data) { return Arr(data, mkx::Shape{static_cast<int64_t>(data.size())}); }

} // namespace

TEST_CASE("vmap(複数入出力): add/multiply/sum/matmulの結果が参照ループと一致する") {
  const int64_t B = 5, n = 4, k = 3;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> a_data(static_cast<size_t>(B * n)), b_data(static_cast<size_t>(B * n)), m_data(static_cast<size_t>(B * k * k));
  for(auto& v : a_data) v = dist(rng);
  for(auto& v : b_data) v = dist(rng);
  for(auto& v : m_data) v = dist(rng);

  Arr a_flat = make_flat(a_data);
  Arr b_flat = make_flat(b_data);
  Arr m_flat = make_flat(m_data);

  // vmap版: batch軸を先頭に持つ実shapeで1回だけfnを呼ぶ。
  Arr a_batched(a_data, mkx::Shape{B, n});
  Arr b_batched(b_data, mkx::Shape{B, n});
  Arr m_batched(m_data, mkx::Shape{B, k * k});

  std::function<std::vector<Arr>(const std::vector<Arr>&)> fn = [k](const std::vector<Arr>& in) { return compute_fn(in, k); };
  auto vmapped                                                = mkx::vmap<VulkanBackend>(fn, {a_batched, b_batched, m_batched});
  mkx::eval<VulkanBackend>(vmapped[0], vmapped[1], vmapped[2]);
  auto v_c  = vmapped[0].to_vector();
  auto v_s  = vmapped[1].to_vector();
  auto v_mm = vmapped[2].to_vector();

  // 参照: 素朴なslice-loop版。
  auto ref = reference_loop({a_flat, b_flat, m_flat}, B, n, k);
  mkx::eval<VulkanBackend>(ref[0], ref[1], ref[2]);
  auto r_c  = ref[0].to_vector();
  auto r_s  = ref[1].to_vector();
  auto r_mm = ref[2].to_vector();

  REQUIRE(v_c.size() == r_c.size());
  for(size_t i = 0; i < v_c.size(); ++i) CHECK(v_c[i] == doctest::Approx(r_c[i]));
  REQUIRE(v_s.size() == r_s.size());
  for(size_t i = 0; i < v_s.size(); ++i) CHECK(v_s[i] == doctest::Approx(r_s[i]));
  REQUIRE(v_mm.size() == r_mm.size());
  for(size_t i = 0; i < v_mm.size(); ++i) CHECK(v_mm[i] == doctest::Approx(r_mm[i]).epsilon(0.001));
}

TEST_CASE("vmap: reduce_max/argmax/argminが参照ループと一致する") {
  const int64_t B = 4, n = 6;
  std::vector<float> data = {
    3, 1, 4, 1, 5, 9, 2, 7, 1, 8, 2, 8, 9, 9, 9, 0, 1, 2, 5, 4, 3, 2, 1, 6,
  };
  Arr batched(data, mkx::Shape{B, n});

  mkx::VmapGuard guard(B, 0);
  auto mx_ = mkx::reduce_max<float, 1, VulkanBackend>(batched);
  auto am  = mkx::argmax<float, 1, VulkanBackend>(batched);
  auto amn = mkx::argmin<float, 1, VulkanBackend>(batched);
  mkx::eval<VulkanBackend>(mx_, am, amn);
  auto v_mx  = mx_.to_vector();
  auto v_am  = am.to_vector();
  auto v_amn = amn.to_vector();

  std::vector<float> expect_mx  = {9, 8, 9, 6};
  std::vector<float> expect_am  = {5, 3, 0, 5};
  std::vector<float> expect_amn = {1, 2, 3, 4};
  for(int64_t e = 0; e < B; ++e) {
    CHECK(v_mx[static_cast<size_t>(e)] == doctest::Approx(expect_mx[static_cast<size_t>(e)]));
    CHECK(v_am[static_cast<size_t>(e)] == doctest::Approx(expect_am[static_cast<size_t>(e)]));
    CHECK(v_amn[static_cast<size_t>(e)] == doctest::Approx(expect_amn[static_cast<size_t>(e)]));
  }
}

TEST_CASE("vmap: cholesky/solve_triangularが参照ループと一致する") {
  const int64_t B = 3, k = 3;
  // 各envで異なる正定値行列(対角優位)+rhs
  std::vector<float> a_data, rhs_data;
  for(int64_t e = 0; e < B; ++e) {
    float scale          = 1.0f + static_cast<float>(e);
    std::vector<float> A = {
      4 * scale, 1, 0, 1, 3 * scale, 1, 0, 1, 2 * scale,
    };
    a_data.insert(a_data.end(), A.begin(), A.end());
    rhs_data.push_back(1.0f * scale);
    rhs_data.push_back(2.0f);
    rhs_data.push_back(3.0f);
  }

  Arr a_flat(a_data, mkx::Shape{static_cast<int64_t>(a_data.size())});
  Arr rhs_flat(rhs_data, mkx::Shape{static_cast<int64_t>(rhs_data.size())});
  Arr a_batched(a_data, mkx::Shape{B, k, k});
  Arr rhs_batched(rhs_data, mkx::Shape{B, k});

  auto chol_ref_single = [k](const Arr& a2, const Arr& rhs2) {
    auto m = mkx::reshape<float, 1, 2, VulkanBackend>(a2, mkx::Shape{k, k});
    auto l = mkx::cholesky<float, VulkanBackend>(m);
    return mkx::solve_triangular<float, VulkanBackend>(l, rhs2);
  };

  std::vector<Arr> ref_pieces;
  for(int64_t e = 0; e < B; ++e) {
    Arr a2   = mkx::reshape<float, 1, 1, VulkanBackend>(mkx::slice(a_flat, {e * k * k}, {(e + 1) * k * k}), mkx::Shape{k * k});
    Arr rhs2 = mkx::reshape<float, 1, 1, VulkanBackend>(mkx::slice(rhs_flat, {e * k}, {(e + 1) * k}), mkx::Shape{k});
    ref_pieces.push_back(chol_ref_single(a2, rhs2));
  }
  Arr ref_acc = ref_pieces[0];
  for(size_t i = 1; i < ref_pieces.size(); ++i) ref_acc = mkx::concatenate(ref_acc, ref_pieces[i], 0);
  mkx::eval<VulkanBackend>(ref_acc);
  auto r = ref_acc.to_vector();

  mkx::VmapGuard guard(B, 0);
  auto l             = mkx::cholesky<float, VulkanBackend>(mkx::array<float, 2, VulkanBackend>(a_batched.node()));
  Arr batched_result = mkx::solve_triangular<float, VulkanBackend>(l, mkx::array<float, 1, VulkanBackend>(rhs_batched.node()));
  mkx::eval<VulkanBackend>(batched_result);
  auto v = batched_result.to_vector();

  REQUIRE(v.size() == r.size());
  for(size_t i = 0; i < v.size(); ++i) CHECK(v[i] == doctest::Approx(r[i]).epsilon(0.001));
}

TEST_CASE("vmap: batch_sizeを増やしても実行時間が超線形に増えない") {
  const int64_t n = 8, k = 3;

  auto run = [&](int64_t B) {
    std::vector<float> a_data(static_cast<size_t>(B * n), 1.0f), b_data(static_cast<size_t>(B * n), 2.0f), m_data(static_cast<size_t>(B * k * k), 0.5f);
    Arr a_batched(a_data, mkx::Shape{B, n});
    Arr b_batched(b_data, mkx::Shape{B, n});
    Arr m_batched(m_data, mkx::Shape{B, k * k});
    std::function<std::vector<Arr>(const std::vector<Arr>&)> fn = [k](const std::vector<Arr>& in) { return compute_fn(in, k); };

    auto t0  = std::chrono::steady_clock::now();
    auto out = mkx::vmap<VulkanBackend>(fn, {a_batched, b_batched, m_batched});
    mkx::eval<VulkanBackend>(out[0], out[1], out[2]);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
  };

  run(32); // warmup(pipeline compile込み)
  double t_small = run(64);
  double t_large = run(512); // 8倍のbatch_size

  // グラフ複製が起きていれば8倍を大きく超えるはず。定数オーバーヘッドも見込み緩い上限にする。
  CHECK(t_large < t_small * 20.0);
}
