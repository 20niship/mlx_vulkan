#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/ops/shape.hpp>
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

mkx::array<float, 1> const1(float v, int64_t n) { return make1(std::vector<float>(static_cast<size_t>(n), v)); }
} // namespace

// MuJoCo-MLX-Cppのmkx_qnorm(kinematics.hpp)相当: sqrt(sum(square(v)))をwhereでゼロ割保護しつつ正規化する複合グラフ。
TEST_CASE("複合グラフ: ベクトル正規化(square+sum+sqrt+where+broadcast)") {
  auto v      = make1({3, 4});
  auto sq     = mkx::square(v);
  auto sum_sq = mkx::sum(sq);
  auto norm   = mkx::sqrt(sum_sq);

  auto eps         = const1(1e-6f, 1);
  auto too_small   = mkx::less(norm, eps);
  auto safe_norm   = mkx::where(too_small, eps, norm);
  auto safe_norm_b = mkx::broadcast_to(safe_norm, mkx::Shape{2});
  auto normalized  = mkx::divide(v, safe_norm_b);

  mkx::eval<VulkanBackend>(normalized);
  auto out = normalized.to_vector<VulkanBackend>();

  float cpu_norm = std::sqrt(3.0f * 3.0f + 4.0f * 4.0f);
  REQUIRE(out.size() == 2);
  CHECK(out[0] == doctest::Approx(3.0f / cpu_norm));
  CHECK(out[1] == doctest::Approx(4.0f / cpu_norm));
}

// 衝突距離計算のような「sqrt(distsq)をclipでmax距離にクランプする」パターン。
TEST_CASE("複合グラフ: 距離計算+clip(sub+square+sum+sqrt+clip)") {
  auto p1     = make1({0, 0, 0});
  auto p2     = make1({3, 4, 0});
  auto diff   = mkx::subtract(p2, p1);
  auto distsq = mkx::sum(mkx::square(diff));
  auto dist   = mkx::sqrt(distsq);

  auto lo      = const1(0.0f, 1);
  auto hi      = const1(4.0f, 1);
  auto clipped = mkx::clip(dist, lo, hi);

  mkx::eval<VulkanBackend>(dist, clipped);
  float cpu_dist = std::sqrt(3.0f * 3.0f + 4.0f * 4.0f + 0.0f * 0.0f);
  CHECK(dist.to_vector<VulkanBackend>()[0] == doctest::Approx(cpu_dist));
  CHECK(clipped.to_vector<VulkanBackend>()[0] == doctest::Approx(std::min(cpu_dist, 4.0f)));
}

// 運動エネルギー計算(0.5*sum(v^2))をmax capでclipするパターン。reduction+elementwiseの混在。
TEST_CASE("複合グラフ: 運動エネルギー計算+上限clip(square+sum+mul+clip)") {
  auto v      = make1({1, 2, 3});
  auto energy = mkx::multiply(mkx::sum(mkx::square(v)), const1(0.5f, 1));

  auto lo     = const1(0.0f, 1);
  auto hi     = const1(5.0f, 1);
  auto capped = mkx::clip(energy, lo, hi);

  mkx::eval<VulkanBackend>(energy, capped);
  float cpu_energy = 0.5f * (1 * 1 + 2 * 2 + 3 * 3);
  CHECK(energy.to_vector<VulkanBackend>()[0] == doctest::Approx(cpu_energy));
  CHECK(capped.to_vector<VulkanBackend>()[0] == doctest::Approx(std::min(cpu_energy, 5.0f)));
}

namespace {
// eval.hppのCpuFallback実装とは独立に、テスト側で同じアルゴリズムをCPU参照実装として持ちGPUグラフの結果と突き合わせる。
std::vector<float> ref_cholesky(const std::vector<float>& a, int n) {
  std::vector<float> L(static_cast<size_t>(n * n), 0.0f);
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j <= i; ++j) {
      float sum = a[static_cast<size_t>(i * n + j)];
      for(int k = 0; k < j; ++k) sum -= L[static_cast<size_t>(i * n + k)] * L[static_cast<size_t>(j * n + k)];
      L[static_cast<size_t>(i * n + j)] = (i == j) ? std::sqrt(sum) : sum / L[static_cast<size_t>(j * n + j)];
    }
  }
  return L;
}

std::vector<float> ref_forward_substitute(const std::vector<float>& L, const std::vector<float>& b, int n) {
  std::vector<float> x(static_cast<size_t>(n));
  for(int i = 0; i < n; ++i) {
    float sum = b[static_cast<size_t>(i)];
    for(int j = 0; j < i; ++j) sum -= L[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
    x[static_cast<size_t>(i)] = sum / L[static_cast<size_t>(i * n + i)];
  }
  return x;
}
} // namespace

// CPUフォールバック(Cholesky/SolveTriangular)の出力をGPU elementwiseグラフ(v_new=v0+a*dt)へ流し込む複合パイプライン。
TEST_CASE("複合グラフ: CPUフォールバック(cholesky+solve_triangular)からGPU積分へ流し込む") {
  std::vector<float> mass    = {4, 2, 2, 3}; // 対称正定値
  std::vector<float> force   = {1, 1};
  float dt                   = 0.1f;
  std::vector<float> v0_data = {0, 0};

  auto m = mkx::reshape<float, 1, 2>(make1(mass), mkx::Shape{2, 2});
  auto l = mkx::cholesky(m);
  auto f = make1(force);
  auto a = mkx::solve_triangular(l, f);

  auto v0     = make1(v0_data);
  auto dt_arr = const1(dt, 2);
  auto v_new  = mkx::add(v0, mkx::multiply(a, dt_arr));

  auto vmax      = const1(100.0f, 2);
  auto vmin      = const1(-100.0f, 2);
  auto v_clamped = mkx::clip(v_new, vmin, vmax);

  mkx::eval<VulkanBackend>(v_clamped);
  auto gpu_v = v_clamped.to_vector<VulkanBackend>();

  auto ref_l               = ref_cholesky(mass, 2);
  auto ref_a               = ref_forward_substitute(ref_l, force, 2);
  std::vector<float> ref_v = {v0_data[0] + ref_a[0] * dt, v0_data[1] + ref_a[1] * dt};

  REQUIRE(gpu_v.size() == 2);
  CHECK(gpu_v[0] == doctest::Approx(ref_v[0]));
  CHECK(gpu_v[1] == doctest::Approx(ref_v[1]));
}

namespace {
std::vector<float> ref_matmul(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
  std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
  for(int i = 0; i < m; ++i) {
    for(int j = 0; j < n; ++j) {
      float s = 0.0f;
      for(int p = 0; p < k; ++p) s += a[static_cast<size_t>(i * k + p)] * b[static_cast<size_t>(p * n + j)];
      c[static_cast<size_t>(i * n + j)] = s;
    }
  }
  return c;
}
} // namespace

// 8x16 * 16x8の行列積を行方向でsum_axis+broadcast_to+divideで正規化する複合グラフ。matmul単体テストより大きいサイズで検証する。
TEST_CASE("複合グラフ: 大きめ行列積(8x16*16x8)を行方向で正規化(matmul+sum_axis+broadcast_to+divide)") {
  const int M = 8, K = 16, N = 8;
  std::vector<float> a_data(static_cast<size_t>(M * K));
  std::vector<float> b_data(static_cast<size_t>(K * N));
  for(size_t i = 0; i < a_data.size(); ++i) a_data[i] = static_cast<float>(i % 7) + 1.0f;
  for(size_t i = 0; i < b_data.size(); ++i) b_data[i] = static_cast<float>(i % 5) + 1.0f;

  auto a           = mkx::reshape<float, 1, 2>(make1(a_data), mkx::Shape{M, K});
  auto b           = mkx::reshape<float, 1, 2>(make1(b_data), mkx::Shape{K, N});
  auto c           = mkx::matmul(a, b);
  auto row_sum     = mkx::sum_axis(c, 1);
  auto row_sum_col = mkx::reshape<float, 1, 2>(row_sum, mkx::Shape{M, 1});
  auto row_sum_b   = mkx::broadcast_to(row_sum_col, mkx::Shape{M, N});
  auto normalized  = mkx::divide(c, row_sum_b);

  mkx::eval<VulkanBackend>(normalized);
  auto gpu = normalized.to_vector<VulkanBackend>();

  auto ref_c = ref_matmul(a_data, b_data, M, K, N);
  REQUIRE(gpu.size() == static_cast<size_t>(M * N));
  for(int i = 0; i < M; ++i) {
    float ref_row_sum = 0.0f;
    for(int j = 0; j < N; ++j) ref_row_sum += ref_c[static_cast<size_t>(i * N + j)];
    for(int j = 0; j < N; ++j) {
      float expected = ref_c[static_cast<size_t>(i * N + j)] / ref_row_sum;
      CHECK(gpu[static_cast<size_t>(i * N + j)] == doctest::Approx(expected));
    }
  }
}

// tril(M)+triu(M)-diag(diag(M)) == M という恒等式(対角成分の二重加算を補正)を10x10行列で検証する。
TEST_CASE("複合グラフ: tril+triu-diagの恒等式(10x10)") {
  const int n = 10;
  std::vector<float> m_data(static_cast<size_t>(n * n));
  for(size_t i = 0; i < m_data.size(); ++i) m_data[i] = static_cast<float>(i % 13) - 6.0f;

  auto m  = mkx::reshape<float, 1, 2>(make1(m_data), mkx::Shape{n, n});
  auto lo = mkx::tril(m);
  auto up = mkx::triu(m);

  std::vector<float> diag_vals(static_cast<size_t>(n));
  for(int i = 0; i < n; ++i) diag_vals[static_cast<size_t>(i)] = m_data[static_cast<size_t>(i * n + i)];
  auto d = mkx::diag(make1(diag_vals));

  auto reconstructed  = mkx::subtract(mkx::add(lo, up), d);
  auto diff           = mkx::subtract(reconstructed, m);
  auto total_abs_diff = mkx::sum(mkx::abs(mkx::flatten(diff)));

  mkx::eval<VulkanBackend>(reconstructed, total_abs_diff);
  auto gpu_recon = reconstructed.to_vector<VulkanBackend>();
  CHECK(total_abs_diff.to_vector<VulkanBackend>()[0] == doctest::Approx(0.0f));
  for(size_t i = 0; i < m_data.size(); ++i) CHECK(gpu_recon[i] == doctest::Approx(m_data[i]));
}

// 12x12対角優位SPD行列でCholesky+SolveTriangular(CPUフォールバック)し、出力をsign(a)*power(abs(a),2)=a*|a|(ダンピング項の定番式)に通す。
TEST_CASE("複合グラフ: 大きめCholesky(12x12)+CPUフォールバック出力をpower/sign連鎖に通す") {
  const int n = 12;
  std::vector<float> m_data(static_cast<size_t>(n * n), 0.0f);
  for(int i = 0; i < n; ++i) {
    float off_sum = 0.0f;
    for(int j = 0; j < n; ++j) {
      if(i == j) continue;
      float v                                = static_cast<float>((i * 3 + j) % 5) * 0.1f;
      m_data[static_cast<size_t>(i * n + j)] = v;
      off_sum += std::abs(v);
    }
    m_data[static_cast<size_t>(i * n + i)] = off_sum + static_cast<float>(i + 1);
  }
  std::vector<float> force(static_cast<size_t>(n));
  for(int i = 0; i < n; ++i) force[static_cast<size_t>(i)] = static_cast<float>(i + 1) * 0.5f;

  auto m = mkx::reshape<float, 1, 2>(make1(m_data), mkx::Shape{n, n});
  auto l = mkx::cholesky(m);
  auto f = make1(force);
  auto a = mkx::solve_triangular(l, f);

  auto damped = mkx::multiply(mkx::sign(a), mkx::power(mkx::abs(a), const1(2.0f, n)));

  mkx::eval<VulkanBackend>(damped);
  auto gpu = damped.to_vector<VulkanBackend>();

  auto ref_l = ref_cholesky(m_data, n);
  auto ref_a = ref_forward_substitute(ref_l, force, n);
  REQUIRE(gpu.size() == static_cast<size_t>(n));
  for(int i = 0; i < n; ++i) {
    float expected = (ref_a[static_cast<size_t>(i)] >= 0.0f ? 1.0f : -1.0f) * std::pow(std::abs(ref_a[static_cast<size_t>(i)]), 2.0f);
    CHECK(gpu[static_cast<size_t>(i)] == doctest::Approx(expected));
  }
}

// 4dofのバネ+ダンパ+力制限+半陰的積分+ポテンシャルエネルギーを1グラフで計算する(18演算呼び出し)。
TEST_CASE("複合グラフ: バネダンパ力制限積分(clip/negative/multiply/add/abs/sign/where等18演算)") {
  const int n               = 4;
  std::vector<float> x_data = {1.5f, -2.5f, 0.3f, 3.0f};
  std::vector<float> v_data = {0.5f, -0.2f, 1.0f, -1.5f};
  float k = 50.0f, c = 4.0f, xlim = 2.0f, force_limit = 60.0f, dt = 0.02f;

  auto x          = make1(x_data);
  auto v          = make1(v_data);
  auto xlim_lo    = const1(-xlim, n);
  auto xlim_hi    = const1(xlim, n);
  auto k_arr      = const1(k, n);
  auto c_arr      = const1(c, n);
  auto limit_arr  = const1(force_limit, n);
  auto dt_arr     = const1(dt, n);
  auto half_k_arr = const1(0.5f * k, n);

  auto clipped_x        = mkx::clip(x, xlim_lo, xlim_hi);
  auto spring_raw       = mkx::multiply(clipped_x, k_arr);
  auto spring_force     = mkx::negative(spring_raw);
  auto damp_raw         = mkx::multiply(v, c_arr);
  auto damp_force       = mkx::negative(damp_raw);
  auto total_force      = mkx::add(spring_force, damp_force);
  auto abs_force        = mkx::abs(total_force);
  auto over_limit       = mkx::greater(abs_force, limit_arr);
  auto force_sign       = mkx::sign(total_force);
  auto capped_force     = mkx::multiply(force_sign, limit_arr);
  auto safe_force       = mkx::where(over_limit, capped_force, total_force);
  auto dv               = mkx::multiply(safe_force, dt_arr);
  auto new_v            = mkx::add(v, dv);
  auto dx               = mkx::multiply(new_v, dt_arr);
  auto new_x            = mkx::add(x, dx);
  auto x2               = mkx::square(clipped_x);
  auto pe_terms         = mkx::multiply(x2, half_k_arr);
  auto potential_energy = mkx::sum(pe_terms);

  mkx::eval<VulkanBackend>(new_x, new_v, potential_energy);
  auto gpu_x   = new_x.to_vector<VulkanBackend>();
  auto gpu_v   = new_v.to_vector<VulkanBackend>();
  float gpu_pe = potential_energy.to_vector<VulkanBackend>()[0];

  float ref_pe = 0.0f;
  for(int i = 0; i < n; ++i) {
    float cx     = std::clamp(x_data[static_cast<size_t>(i)], -xlim, xlim);
    float spring = -cx * k;
    float damp   = -v_data[static_cast<size_t>(i)] * c;
    float total  = spring + damp;
    float safe   = (std::abs(total) > force_limit) ? (total >= 0.0f ? 1.0f : -1.0f) * force_limit : total;
    float nv     = v_data[static_cast<size_t>(i)] + safe * dt;
    float nx     = x_data[static_cast<size_t>(i)] + nv * dt;
    CHECK(gpu_v[static_cast<size_t>(i)] == doctest::Approx(nv));
    CHECK(gpu_x[static_cast<size_t>(i)] == doctest::Approx(nx));
    ref_pe += 0.5f * k * cx * cx;
  }
  CHECK(gpu_pe == doctest::Approx(ref_pe));
}

// 角度theta回転をelementwiseで組み立て、ノルム計算+安全な正規化までを1グラフで行う(17演算呼び出し)。
TEST_CASE("複合グラフ: 回転+ノルム計算+安全正規化(sin/cos/square/sqrt/clip/maximum/divide等17演算)") {
  const int n                   = 4;
  std::vector<float> theta_data = {0.3f, 1.1f, -0.7f, 2.0f};
  std::vector<float> x_data     = {1.0f, 2.0f, 0.5f, -1.5f};
  std::vector<float> y_data     = {0.5f, -1.0f, 1.5f, 2.0f};

  auto theta = make1(theta_data);
  auto x     = make1(x_data);
  auto y     = make1(y_data);
  auto zero  = const1(0.0f, n);
  auto maxr  = const1(3.0f, n);
  auto eps   = const1(1e-6f, n);

  auto ct            = mkx::cos(theta);
  auto st            = mkx::sin(theta);
  auto t1            = mkx::multiply(x, ct);
  auto t2            = mkx::multiply(y, st);
  auto rx            = mkx::subtract(t1, t2);
  auto t3            = mkx::multiply(x, st);
  auto t4            = mkx::multiply(y, ct);
  auto ry            = mkx::add(t3, t4);
  auto rx2           = mkx::square(rx);
  auto ry2           = mkx::square(ry);
  auto rsq           = mkx::add(rx2, ry2);
  auto rnorm         = mkx::sqrt(rsq);
  auto rnorm_clamped = mkx::clip(rnorm, zero, maxr);
  auto safe_norm     = mkx::maximum(rnorm_clamped, eps);
  auto unit_x        = mkx::divide(rx, safe_norm);
  auto unit_y        = mkx::divide(ry, safe_norm);

  mkx::eval<VulkanBackend>(unit_x, unit_y, rnorm);
  auto gpu_ux   = unit_x.to_vector<VulkanBackend>();
  auto gpu_uy   = unit_y.to_vector<VulkanBackend>();
  auto gpu_norm = rnorm.to_vector<VulkanBackend>();

  for(int i = 0; i < n; ++i) {
    float c        = std::cos(theta_data[static_cast<size_t>(i)]);
    float s        = std::sin(theta_data[static_cast<size_t>(i)]);
    float rx_ref   = x_data[static_cast<size_t>(i)] * c - y_data[static_cast<size_t>(i)] * s;
    float ry_ref   = x_data[static_cast<size_t>(i)] * s + y_data[static_cast<size_t>(i)] * c;
    float norm_ref = std::sqrt(rx_ref * rx_ref + ry_ref * ry_ref);
    CHECK(gpu_norm[static_cast<size_t>(i)] == doctest::Approx(norm_ref));
    float clamped = std::min(norm_ref, 3.0f);
    float safe    = std::max(clamped, 1e-6f);
    CHECK(gpu_ux[static_cast<size_t>(i)] == doctest::Approx(rx_ref / safe));
    CHECK(gpu_uy[static_cast<size_t>(i)] == doctest::Approx(ry_ref / safe));
  }
}

// reduce_max/argmax/takeの整合性と比較・論理演算のマスク合成を1グラフで検証する(13演算呼び出し)。
TEST_CASE("複合グラフ: argmax/take整合性+比較・論理マスク合成(reduce_max/take/equal/logical_and等13演算)") {
  std::vector<float> values_data = {3.0f, 7.0f, -1.0f, 7.0f, 2.0f, 5.0f};
  const int n                    = static_cast<int>(values_data.size());
  auto values                    = make1(values_data);
  auto thresh1                   = const1(4.0f, n);
  auto thresh2                   = const1(0.0f, n);

  auto vmax         = mkx::reduce_max(values);
  auto idx          = mkx::argmax(values);
  auto vmax_b       = mkx::broadcast_to(vmax, mkx::Shape{n});
  auto is_max_mask  = mkx::equal(values, vmax_b);
  auto above_thresh = mkx::greater(values, thresh1);
  auto below2       = mkx::less(values, thresh2);
  auto combined1    = mkx::logical_and(above_thresh, is_max_mask);
  auto combined2    = mkx::logical_or(combined1, below2);
  auto inverted     = mkx::logical_not(combined2);
  auto count_max    = mkx::sum(is_max_mask);
  auto gathered_max = mkx::take(values, idx, 0);
  auto diff         = mkx::subtract(gathered_max, vmax);
  auto abs_diff     = mkx::sum(mkx::abs(diff));

  mkx::eval<VulkanBackend>(count_max, gathered_max, abs_diff, inverted);
  float ref_max     = *std::max_element(values_data.begin(), values_data.end());
  int ref_count_max = static_cast<int>(std::count(values_data.begin(), values_data.end(), ref_max));

  CHECK(count_max.to_vector<VulkanBackend>()[0] == doctest::Approx(static_cast<float>(ref_count_max)));
  CHECK(gathered_max.to_vector<VulkanBackend>()[0] == doctest::Approx(ref_max));
  CHECK(abs_diff.to_vector<VulkanBackend>()[0] == doctest::Approx(0.0f));

  auto inv = inverted.to_vector<VulkanBackend>();
  for(int i = 0; i < n; ++i) {
    bool ref_above    = values_data[static_cast<size_t>(i)] > 4.0f;
    bool ref_is_max   = values_data[static_cast<size_t>(i)] == ref_max;
    bool ref_below2   = values_data[static_cast<size_t>(i)] < 0.0f;
    bool ref_combined = (ref_above && ref_is_max) || ref_below2;
    CHECK(inv[static_cast<size_t>(i)] == doctest::Approx(ref_combined ? 0.0f : 1.0f));
  }
}

// A*A^Tの対称性検証とslice+concatenateによる再構築を1グラフで行う(transpose/matmul/tril/triu/slice/concatenate等16演算)。
TEST_CASE("複合グラフ: A*A^T対称性+slice/concatenate再構築(transpose/matmul/tril/triu/slice/concat等16演算)") {
  const int M = 3, K = 4;
  std::vector<float> a_data(static_cast<size_t>(M * K));
  for(size_t i = 0; i < a_data.size(); ++i) a_data[i] = static_cast<float>(i % 6) - 2.5f;

  auto a            = mkx::reshape<float, 1, 2>(make1(a_data), mkx::Shape{M, K});
  auto at           = mkx::transpose(a, {1, 0});
  auto c            = mkx::matmul(a, at);
  auto ct           = mkx::transpose(c, {1, 0});
  auto sym_diff     = mkx::subtract(c, ct);
  auto abs_sym_diff = mkx::abs(mkx::flatten(sym_diff));
  auto total_asym   = mkx::sum(abs_sym_diff);

  auto lo       = mkx::tril(c);
  auto up       = mkx::triu(c);
  auto combined = mkx::add(lo, up);

  auto top           = mkx::slice(c, {0, 0}, {2, static_cast<int64_t>(M)});
  auto bottom        = mkx::slice(c, {2, 0}, {3, static_cast<int64_t>(M)});
  auto reconstructed = mkx::concatenate(top, bottom, 0);
  auto diff2         = mkx::subtract(reconstructed, c);
  auto total_diff2   = mkx::sum(mkx::abs(mkx::flatten(diff2)));

  mkx::eval<VulkanBackend>(c, combined, total_asym, total_diff2);
  auto gpu_c        = c.to_vector<VulkanBackend>();
  auto gpu_combined = combined.to_vector<VulkanBackend>();

  std::vector<float> ref_c(static_cast<size_t>(M * M), 0.0f);
  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < M; ++j) {
      float s = 0.0f;
      for(int p = 0; p < K; ++p) {
        s += a_data[static_cast<size_t>(i * K + p)] * a_data[static_cast<size_t>(j * K + p)];
      }
      ref_c[static_cast<size_t>(i * M + j)] = s;
    }
  }
  REQUIRE(gpu_c.size() == ref_c.size());
  for(size_t i = 0; i < ref_c.size(); ++i) CHECK(gpu_c[i] == doctest::Approx(ref_c[i]));

  CHECK(total_asym.to_vector<VulkanBackend>()[0] == doctest::Approx(0.0f));
  CHECK(total_diff2.to_vector<VulkanBackend>()[0] == doctest::Approx(0.0f));

  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < M; ++j) {
      // 対角はtril/triu両方に含まれるため combined の対角成分は2*C[i][i]になる(意図した仕様)。
      float expected = (j <= i ? ref_c[static_cast<size_t>(i * M + j)] : 0.0f) + (j >= i ? ref_c[static_cast<size_t>(i * M + j)] : 0.0f);
      CHECK(gpu_combined[static_cast<size_t>(i * M + j)] == doctest::Approx(expected));
    }
  }
}
