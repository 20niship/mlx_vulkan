#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
  CustomKernel,       // 所有者: 実際にdispatchし、全出力バッファをmulti_outputsに持つ
  CustomKernelOutput, // 別名: 所有者ノードのmulti_outputs[output_index]を指すだけ
};

// dispatch時にどのGLSLテンプレート群を使うかの分類。
enum class ShaderGroup {
  View,        // shader不要、gpu_bufferを入力から共有するだけ
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
    case ShaderGroup::Custom: return -1; // eval_node側で個別処理するため未使用
  }
  return 2;
}

struct OpNode {
  OpType type;
  Shape shape;
  Dtype dtype;
  std::vector<std::shared_ptr<OpNode>> inputs;
  std::vector<std::byte> imm_data; // スカラー定数・形状メタデータ(push constant化)

  void* gpu_buffer = nullptr; // eval後のバッファハンドル(backend固有、void*で抽象化)
  bool evaluated   = false;

  // 将来のgrad対応用に予約。Phase0-2では未使用、常にnullptr。
  void* backward_fn = nullptr;

  // Phase7 CustomKernel所有者ノード専用(それ以外は未使用のまま)
  std::string custom_source;
  std::vector<Shape> custom_output_shapes;
  std::vector<Dtype> custom_output_dtypes;
  std::array<uint32_t, 3> custom_groups{1, 1, 1};
  std::vector<void*> multi_outputs;

  // Phase7 CustomKernelOutput(別名ノード)専用: 所有者のmulti_outputsのindex
  int output_index = -1;

  // gpu_bufferはbackend非依存void*で型消去されているため解放はBackendを知るeval_node側がここに設定するコールバックで行う。View/CustomKernelOutput(バッファ共有のみ)は未設定のため二重解放にならない。
  std::function<void()> free_gpu_buffer;
  ~OpNode() { if(free_gpu_buffer) free_gpu_buffer(); }
};

using NodePtr = std::shared_ptr<OpNode>;

inline NodePtr make_node(OpType type, Shape shape, Dtype dtype, std::vector<NodePtr> inputs = {}, std::vector<std::byte> imm_data = {}) {
  auto node      = std::make_shared<OpNode>();
  node->type     = type;
  node->shape    = std::move(shape);
  node->dtype    = dtype;
  node->inputs   = std::move(inputs);
  node->imm_data = std::move(imm_data);
  return node;
}

} // namespace mkx
