#pragma once

#include <cstring>
#include <string>
#include <vector>

#include <mkx/core/op_node.hpp>
#include <mkx/shaders/shader_templates.hpp>

namespace mkx {

// 全pipeline共通のpush constant layout(フィールドの用途はShaderGroupごとに違う、shader_templates.hppのpush_decl参照)。
struct Push {
  uint32_t count          = 0;
  float p0                = 0;
  float p1                = 0;
  uint32_t ndim           = 0;
  uint32_t out_shape[4]   = {0, 0, 0, 0};
  uint32_t in_shape[4]    = {0, 0, 0, 0};
  uint32_t in_strides[4]  = {0, 0, 0, 0};
  uint32_t in_base_offset = 0;
};

inline int opcode_for(OpType t) {
  switch(t) {
    case OpType::Fill: return 0;
    case OpType::Range: return 1;
    case OpType::Eye: return 2;
    case OpType::RandomNormal: return 3;

    case OpType::Neg: return 0;
    case OpType::Abs: return 1;
    case OpType::Sqrt: return 2;
    case OpType::Square: return 3;
    case OpType::Sign: return 4;
    case OpType::Floor: return 5;
    case OpType::Sin: return 6;
    case OpType::Cos: return 7;
    case OpType::LogicalNot: return 8;
    case OpType::AsType: return 9;
    case OpType::Copy: return 9;
    case OpType::Diag: return 10;
    case OpType::Tril: return 11;
    case OpType::Triu: return 12;

    case OpType::Add: return 0;
    case OpType::Sub: return 1;
    case OpType::Mul: return 2;
    case OpType::Div: return 3;
    case OpType::Power: return 4;
    case OpType::Equal: return 5;
    case OpType::Greater: return 6;
    case OpType::GreaterEqual: return 7;
    case OpType::Less: return 8;
    case OpType::LessEqual: return 9;
    case OpType::LogicalAnd: return 10;
    case OpType::LogicalOr: return 11;
    case OpType::Max: return 12;
    case OpType::Min: return 13;

    case OpType::Where: return 0;
    case OpType::Clip: return 1;
    case OpType::Cross: return 14;

    case OpType::Sum: return 0;
    case OpType::ReduceMax: return 1;
    case OpType::ArgMax: return 2;
    case OpType::ArgMin: return 3;

    case OpType::SumAxis: return 0;
    case OpType::ReduceMaxAxis: return 1;

    default: return 0;
  }
}

inline std::string_view template_for(OpType t) {
  switch(shader_group_for(t)) {
    case ShaderGroup::Creation: return shaders::creation_glsl;
    case ShaderGroup::Unary: return shaders::unary_glsl;
    case ShaderGroup::Ternary: return shaders::ternary_glsl;
    case ShaderGroup::Gather: return shaders::gather_glsl;
    case ShaderGroup::Concat: return shaders::concat_glsl;
    case ShaderGroup::Take: return shaders::take_glsl;
    case ShaderGroup::Reduce: return shaders::reduce_glsl;
    case ShaderGroup::ReduceAxis: return shaders::reduce_axis_glsl;
    case ShaderGroup::MatMul: return shaders::matmul_glsl;
    case ShaderGroup::LinalgSeq: return (t == OpType::Cholesky) ? shaders::cholesky_glsl : shaders::solve_triangular_glsl;
    default: return shaders::binary_glsl;
  }
}

inline std::string shader_source_for(OpType t, Dtype dtype) {
  std::string src = "#version 450\n";
  // AsTypeは入出力でdtypeが異なりうるため単一SCALARマクロでは扱えない、常にfloatにフォールバック(fp16↔fp32変換は将来対応)
  if(t == OpType::AsType) dtype = Dtype::Float32;
  if(dtype == Dtype::Float16) {
    src += "#extension GL_EXT_shader_16bit_storage : require\n";
    src += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    src += "#define SCALAR float16_t\n";
  } else {
    src += "#define SCALAR float\n";
  }
  src += "#define OPCODE ";
  src += std::to_string(opcode_for(t));
  src += "\n";
  src += shaders::push_decl;
  src += template_for(t);
  return src;
}

inline std::vector<std::byte> pack_push(const Push& push) {
  std::vector<std::byte> out(sizeof(Push));
  std::memcpy(out.data(), &push, sizeof(Push));
  return out;
}

// Cholesky/SolveTriangularの入力0(n×n行列)shapeが{B,n,n}ならB、{n,n}なら1(非バッチ)。
template <class Backend> int64_t linalg_seq_batch(const OpNode<Backend>& node) {
  auto& mat_shape = node.inputs[0]->shape;
  return mat_shape.size() > 2 ? mat_shape[0] : 1;
}

template <class Backend> Push build_push(const OpNode<Backend>& node) {
  Push push;
  push.count = static_cast<uint32_t>(shape_size(node.shape));

  switch(shader_group_for(node.type)) {
    case ShaderGroup::Creation:
    case ShaderGroup::Unary:
      if(node.imm_data.size() >= sizeof(float)) {
        std::memcpy(&push.p0, node.imm_data.data(), sizeof(float));
      }
      if(node.imm_data.size() >= sizeof(float) * 2) {
        std::memcpy(&push.p1, node.imm_data.data() + sizeof(float), sizeof(float));
      }
      break;
    case ShaderGroup::Gather:
    case ShaderGroup::Concat:
    case ShaderGroup::Take:
    case ShaderGroup::MatMul:
    case ShaderGroup::ReduceAxis:
      // Gather/Concat/Take/MatMul/ReduceAxis: ops側がpack_push()でPush全体を事前構築しimm_dataに積む
      if(node.imm_data.size() == sizeof(Push)) {
        std::memcpy(&push, node.imm_data.data(), sizeof(Push));
        push.count = static_cast<uint32_t>(shape_size(node.shape));
      }
      break;
    case ShaderGroup::LinalgSeq:
      // n/batchはnode.shapeから自明(Cholesky出力{B,n,n}/{n,n}、SolveTriangular出力{B,n}/{n})なのでimm_data不要。
      push.in_base_offset = static_cast<uint32_t>(node.shape.back());
      push.count          = static_cast<uint32_t>(linalg_seq_batch(node));
      break;
    default: break;
  }

  return push;
}

} // namespace mkx
