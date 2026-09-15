#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/types.hpp>

namespace mkx {

enum class OpType {
  Const,
  Fill,
  Range,
  Eye,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Abs,
  Sqrt,
  Square,
  Sign,
  Floor,
  Sin,
  Cos,
  Power,
  Equal,
  Greater,
  GreaterEqual,
  Less,
  LessEqual,
  LogicalAnd,
  LogicalOr,
  LogicalNot,
  Where,
  Clip,
  Max,
  Min,
  AsType,

  // Phase3: 形状操作
  Reshape,
  Flatten,
  Transpose,
  BroadcastTo,
  Tile,
  Slice,
  Concatenate,
  Stack,
  Take,
  Diag,
  Tril,
  Triu,
  Copy,

  // Phase4: リダクション・線形代数
  Sum,
  ReduceMax,
  ArgMax,
  ArgMin,
  SumAxis,
  ReduceMaxAxis,
  MatMul,
  Cholesky,
  SolveTriangular,
  Cross,

  // Phase5: 乱数
  RandomNormal,

  // Phase7: mx::fast::metal_kernel互換のカスタムカーネル
  CustomKernel,       // 所有者: 実際にdispatchする。出力は自身のoutput_aliasesが指す別名ノード群
  CustomKernelOutput, // 別名: 所有者ノードのoutput_index番目の出力を指すだけ
};

// dispatch時にどのGLSLテンプレート群を使うかの分類。
enum class ShaderGroup {
  View,        // shader不要、入力のバッファをそのまま共有する
  Creation,    // 入力0、出力のみ
  Unary,       // 入力1、同一index参照 (Diag/Tril/Triu/Copyも同一テンプレートに同居)
  Binary,      // 入力2、同一index参照
  Ternary,     // 入力3、同一index参照
  Gather,      // 入力1、出力indexから入力indexへアフィン+modulo写像 (Transpose/Slice/Tile/BroadcastTo)
  Concat,      // 入力2、axis位置で入力を切り替え (Concatenate/Stack)
  Take,        // 入力2 (data, indices)、axis方向のgather
  Reduce,      // 入力1、単一work-group内リダクション (Sum/ReduceMax/ArgMax/ArgMin, 全体リダクション)
  ReduceAxis,  // 入力1、出力要素ごとに1 work-groupを割り当てる軸指定リダクション (SumAxis/ReduceMaxAxis)
  MatMul,      // 入力2、タイル化GEMM
  CpuFallback, // Cholesky/SolveTriangular: GPU→CPU→GPU
  Custom,      // CustomKernel所有者: 完全自前のGLSLソース+可変長入出力
};

inline ShaderGroup shader_group_for(OpType t) {
  switch(t) {
    case OpType::Reshape:
    case OpType::Flatten: return ShaderGroup::View;
    case OpType::Const:
    case OpType::Fill:
    case OpType::Range:
    case OpType::Eye:
    case OpType::RandomNormal: return ShaderGroup::Creation;
    case OpType::Transpose:
    case OpType::BroadcastTo:
    case OpType::Tile:
    case OpType::Slice: return ShaderGroup::Gather;
    case OpType::Concatenate:
    case OpType::Stack: return ShaderGroup::Concat;
    case OpType::Take: return ShaderGroup::Take;
    case OpType::Where:
    case OpType::Clip: return ShaderGroup::Ternary;
    case OpType::Neg:
    case OpType::Abs:
    case OpType::Sqrt:
    case OpType::Square:
    case OpType::Sign:
    case OpType::Floor:
    case OpType::Sin:
    case OpType::Cos:
    case OpType::LogicalNot:
    case OpType::AsType:
    case OpType::Diag:
    case OpType::Tril:
    case OpType::Triu:
    case OpType::Copy: return ShaderGroup::Unary;
    case OpType::Sum:
    case OpType::ReduceMax:
    case OpType::ArgMax:
    case OpType::ArgMin: return ShaderGroup::Reduce;
    case OpType::SumAxis:
    case OpType::ReduceMaxAxis: return ShaderGroup::ReduceAxis;
    case OpType::MatMul: return ShaderGroup::MatMul;
    case OpType::Cholesky:
    case OpType::SolveTriangular: return ShaderGroup::CpuFallback;
    case OpType::CustomKernel:
    case OpType::CustomKernelOutput: return ShaderGroup::Custom;
    default: return ShaderGroup::Binary;
  }
}

// dispatch時にbindするGPU入力バッファの数(View/CpuFallbackは対象外)。
inline int op_arity(OpType t) {
  switch(shader_group_for(t)) {
    case ShaderGroup::View:
    case ShaderGroup::Creation: return 0;
    case ShaderGroup::Unary:
    case ShaderGroup::Gather:
    case ShaderGroup::Reduce:
    case ShaderGroup::ReduceAxis:
    case ShaderGroup::CpuFallback: return 1;
    case ShaderGroup::Binary:
    case ShaderGroup::Concat:
    case ShaderGroup::Take:
    case ShaderGroup::MatMul: return 2;
    case ShaderGroup::Ternary: return 3;
    case ShaderGroup::Custom: return -1; // eval側で個別処理するため未使用
  }
  return 2;
}

// GPUバッファはBackend側がnode自身のアドレスをキーに管理する(get_or_allocate/release_node)ため、ここにBuffer型は置かない。
template <ComputeBackend Backend> struct OpNode {
  OpType type;
  Shape shape;
  Dtype dtype;
  std::vector<std::shared_ptr<OpNode<Backend>>> inputs;
  std::vector<std::byte> imm_data; // スカラー定数・形状メタデータ(push constant化)

  bool evaluated = false;

  // OpType::Const専用: ホスト側にmemcpyしておいた生データ。evalのタイミングでBackendへupload。
  std::vector<std::byte> host_data;

  // 将来のgrad対応用に予約。Phase0-2では未使用、常にnullptr。
  void* backward_fn = nullptr;

  // Phase7 CustomKernel所有者ノード専用(それ以外は未使用のまま)
  std::string custom_source;
  std::vector<Shape> custom_output_shapes;
  std::vector<Dtype> custom_output_dtypes;
  std::array<uint32_t, 3> custom_groups{1, 1, 1};
  // 所有者から見た各出力の別名ノード(CustomKernelOutput)。グラフ構造情報であり、値そのものは持たない。
  std::vector<std::weak_ptr<OpNode<Backend>>> output_aliases;

  // Phase7 CustomKernelOutput(別名ノード)専用: 所有者(inputs[0])のoutput_aliasesの何番目か
  int output_index = -1;

  // 呼び出し場所(loc_id)+owner単位でバッファを使い回す永続ノード用のID情報(Buffer型そのものは持たない)。
  bool is_permanent            = false;
  uint64_t persistent_loc_id   = 0;
  const void* persistent_owner = nullptr;
};

template <ComputeBackend Backend> using NodePtr = std::shared_ptr<OpNode<Backend>>;

template <ComputeBackend Backend> NodePtr<Backend> make_node(OpType type, Shape shape, Dtype dtype, std::vector<NodePtr<Backend>> inputs = {}, std::vector<std::byte> imm_data = {}) {
  auto* raw     = new OpNode<Backend>();
  raw->type     = type;
  raw->shape    = std::move(shape);
  raw->dtype    = dtype;
  raw->inputs   = std::move(inputs);
  raw->imm_data = std::move(imm_data);
  // バッファ解放はBackend側の責務。node解体時にBackendへ後始末(release_node)を委ねる。
  return NodePtr<Backend>(raw, [](OpNode<Backend>* p) {
    Backend::release_node(p);
    delete p;
  });
}

// Reshape/Flatten(View)はバッファを共有するだけなので、入力を辿って実体を持つノードのBufferを取得する。
template <ComputeBackend Backend> typename Backend::Buffer* buffer_for(const NodePtr<Backend>& n) {
  OpNode<Backend>* cur = n.get();
  while((cur->type == OpType::Reshape || cur->type == OpType::Flatten) && !cur->inputs.empty()) cur = cur->inputs[0].get();
  return Backend::get_or_allocate(cur, static_cast<size_t>(shape_size(cur->shape)) * dtype_size(cur->dtype));
}

inline uint64_t persistent_location_hash(const char* file, int line) { return std::hash<std::string>{}(std::string(file) + ":" + std::to_string(line)); }

// evalする前に呼ぶこと。同じ(loc_id, owner)なら次回以降も同じBufferを使い回す。
template <ComputeBackend Backend> void mark_permanent(const NodePtr<Backend>& node, uint64_t loc_id, const void* owner) {
  node->is_permanent      = true;
  node->persistent_loc_id = loc_id;
  node->persistent_owner  = owner;
}

} // namespace mkx
