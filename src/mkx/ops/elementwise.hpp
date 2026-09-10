#pragma once

#include <mkx/core/array.hpp>
#include <mkx/core/op_node.hpp>

namespace mkx {

namespace detail {
template <class T, size_t N> array<T, N> unary_op(OpType t, const array<T, N>& a) { return array<T, N>(make_node(t, a.shape(), a.dtype(), {a.node()})); }

template <class T, size_t N> array<T, N> binary_op(OpType t, const array<T, N>& a, const array<T, N>& b) { return array<T, N>(make_node(t, a.shape(), a.dtype(), {a.node(), b.node()})); }

template <class T, size_t N> array<T, N> ternary_op(OpType t, const array<T, N>& a, const array<T, N>& b, const array<T, N>& c) { return array<T, N>(make_node(t, a.shape(), a.dtype(), {a.node(), b.node(), c.node()})); }
} // namespace detail

template <class T, size_t N> array<T, N> add(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Add, a, b); }
template <class T, size_t N> array<T, N> subtract(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Sub, a, b); }
template <class T, size_t N> array<T, N> multiply(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Mul, a, b); }
template <class T, size_t N> array<T, N> divide(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Div, a, b); }
template <class T, size_t N> array<T, N> power(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Power, a, b); }
template <class T, size_t N> array<T, N> maximum(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Max, a, b); }
template <class T, size_t N> array<T, N> minimum(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Min, a, b); }

template <class T, size_t N> array<T, N> equal(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Equal, a, b); }
template <class T, size_t N> array<T, N> greater(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Greater, a, b); }
template <class T, size_t N> array<T, N> greater_equal(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::GreaterEqual, a, b); }
template <class T, size_t N> array<T, N> less(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::Less, a, b); }
template <class T, size_t N> array<T, N> less_equal(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::LessEqual, a, b); }
template <class T, size_t N> array<T, N> logical_and(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::LogicalAnd, a, b); }
template <class T, size_t N> array<T, N> logical_or(const array<T, N>& a, const array<T, N>& b) { return detail::binary_op(OpType::LogicalOr, a, b); }

template <class T, size_t N> array<T, N> negative(const array<T, N>& a) { return detail::unary_op(OpType::Neg, a); }
template <class T, size_t N> array<T, N> abs(const array<T, N>& a) { return detail::unary_op(OpType::Abs, a); }
template <class T, size_t N> array<T, N> sqrt(const array<T, N>& a) { return detail::unary_op(OpType::Sqrt, a); }
template <class T, size_t N> array<T, N> square(const array<T, N>& a) { return detail::unary_op(OpType::Square, a); }
template <class T, size_t N> array<T, N> sign(const array<T, N>& a) { return detail::unary_op(OpType::Sign, a); }
template <class T, size_t N> array<T, N> floor(const array<T, N>& a) { return detail::unary_op(OpType::Floor, a); }
template <class T, size_t N> array<T, N> sin(const array<T, N>& a) { return detail::unary_op(OpType::Sin, a); }
template <class T, size_t N> array<T, N> cos(const array<T, N>& a) { return detail::unary_op(OpType::Cos, a); }
template <class T, size_t N> array<T, N> logical_not(const array<T, N>& a) { return detail::unary_op(OpType::LogicalNot, a); }

template <class T, size_t N> array<T, N> where(const array<T, N>& cond, const array<T, N>& x, const array<T, N>& y) { return detail::ternary_op(OpType::Where, cond, x, y); }
template <class T, size_t N> array<T, N> clip(const array<T, N>& x, const array<T, N>& lo, const array<T, N>& hi) { return detail::ternary_op(OpType::Clip, x, lo, hi); }

} // namespace mkx
