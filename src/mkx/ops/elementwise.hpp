#pragma once

#include <mkx/core/array.hpp>
#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>

namespace mkx {

namespace detail {
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> unary_op(OpType t, const array<T, N, Backend>& a) { return array<T, N, Backend>(make_node<Backend>(t, a.shape(), a.dtype(), {a.node()})); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> binary_op(OpType t, const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return array<T, N, Backend>(make_node<Backend>(t, a.shape(), a.dtype(), {a.node(), b.node()})); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> ternary_op(OpType t, const array<T, N, Backend>& a, const array<T, N, Backend>& b, const array<T, N, Backend>& c) { return array<T, N, Backend>(make_node<Backend>(t, a.shape(), a.dtype(), {a.node(), b.node(), c.node()})); }
} // namespace detail

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> add(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Add, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> subtract(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Sub, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> multiply(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Mul, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> divide(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Div, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> power(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Power, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> maximum(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Max, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> minimum(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Min, a, b); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> equal(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Equal, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> greater(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Greater, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> greater_equal(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::GreaterEqual, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> less(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::Less, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> less_equal(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::LessEqual, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> logical_and(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::LogicalAnd, a, b); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> logical_or(const array<T, N, Backend>& a, const array<T, N, Backend>& b) { return detail::binary_op(OpType::LogicalOr, a, b); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> negative(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Neg, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> abs(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Abs, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> sqrt(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Sqrt, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> square(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Square, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> sign(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Sign, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> floor(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Floor, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> sin(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Sin, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> cos(const array<T, N, Backend>& a) { return detail::unary_op(OpType::Cos, a); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> logical_not(const array<T, N, Backend>& a) { return detail::unary_op(OpType::LogicalNot, a); }

template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> where(const array<T, N, Backend>& cond, const array<T, N, Backend>& x, const array<T, N, Backend>& y) { return detail::ternary_op(OpType::Where, cond, x, y); }
template <class T, size_t N, ComputeBackend Backend> array<T, N, Backend> clip(const array<T, N, Backend>& x, const array<T, N, Backend>& lo, const array<T, N, Backend>& hi) { return detail::ternary_op(OpType::Clip, x, lo, hi); }

} // namespace mkx
