#include <doctest/doctest.h>

#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/ops/creation.hpp>
#include <mkx/ops/elementwise.hpp>
#include <mkx/ops/linalg.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

using mkx::VulkanBackend;

TEST_CASE("fp16: addはfloat32相当の精度で一致する") {
  mkx::array<float, 1> a({1.0f, 2.0f, 3.0f}, mkx::Shape{3}, mkx::Dtype::Float16);
  mkx::array<float, 1> b({4.0f, 5.0f, 6.0f}, mkx::Shape{3}, mkx::Dtype::Float16);

  auto r = mkx::add(a, b);
  mkx::eval(r);
  auto v = r.to_vector();
  CHECK(v[0] == doctest::Approx(5.0f).epsilon(0.01));
  CHECK(v[1] == doctest::Approx(7.0f).epsilon(0.01));
  CHECK(v[2] == doctest::Approx(9.0f).epsilon(0.01));
}

TEST_CASE("fp16: multiplyはfloat32相当の精度で一致する") {
  mkx::array<float, 1> a({2.0f, 3.0f, 4.0f}, mkx::Shape{3}, mkx::Dtype::Float16);
  mkx::array<float, 1> b({1.5f, 2.5f, 0.5f}, mkx::Shape{3}, mkx::Dtype::Float16);

  auto r = mkx::multiply(a, b);
  mkx::eval(r);
  auto v = r.to_vector();
  CHECK(v[0] == doctest::Approx(3.0f).epsilon(0.01));
  CHECK(v[1] == doctest::Approx(7.5f).epsilon(0.01));
  CHECK(v[2] == doctest::Approx(2.0f).epsilon(0.01));
}

TEST_CASE("fp16: sumはfloat32相当の精度で一致する") {
  mkx::array<float, 1> a({1.0f, 2.0f, 3.0f, 4.0f}, mkx::Shape{4}, mkx::Dtype::Float16);

  auto r = mkx::sum(a);
  mkx::eval(r);
  auto v = r.to_vector();
  CHECK(v[0] == doctest::Approx(10.0f).epsilon(0.01));
}
