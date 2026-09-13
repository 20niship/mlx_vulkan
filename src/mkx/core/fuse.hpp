#pragma once

// mx::compile()向けoperator fusion(要素単位演算+形状変換のみ、Reduce/MatMul/Take/Concat/CpuFallback/CustomKernel/Reshape/Flattenは従来通りdispatch境界)。

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/eval.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/shaders/shader_source.hpp>

namespace mkx::detail {

inline bool fuse_is_fusable(OpType t) {
  switch(t) {
    case OpType::Add:
    case OpType::Sub:
    case OpType::Mul:
    case OpType::Div:
    case OpType::Power:
    case OpType::Equal:
    case OpType::Greater:
    case OpType::GreaterEqual:
    case OpType::Less:
    case OpType::LessEqual:
    case OpType::LogicalAnd:
    case OpType::LogicalOr:
    case OpType::Max:
    case OpType::Min:
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
    case OpType::Copy:
    case OpType::Where:
    case OpType::Clip:
    case OpType::BroadcastTo:
    case OpType::Tile:
    case OpType::Transpose:
    case OpType::Slice: return true;
    default: return false;
  }
}

// unary_glsl/binary_glsl/ternary_glslの各OPCODE分岐と数値的に一致させる(Diag/Tril/Triu/Crossはfuse_is_fusable側で対象外)。
inline std::string fuse_scalar_expr(OpType t, const std::vector<std::string>& a) {
  switch(t) {
    case OpType::Add: return "(" + a[0] + " + " + a[1] + ")";
    case OpType::Sub: return "(" + a[0] + " - " + a[1] + ")";
    case OpType::Mul: return "(" + a[0] + " * " + a[1] + ")";
    case OpType::Div: return "(" + a[0] + " / " + a[1] + ")";
    case OpType::Power: return "pow(" + a[0] + ", " + a[1] + ")";
    case OpType::Equal: return "((" + a[0] + " == " + a[1] + ") ? 1.0 : 0.0)";
    case OpType::Greater: return "((" + a[0] + " > " + a[1] + ") ? 1.0 : 0.0)";
    case OpType::GreaterEqual: return "((" + a[0] + " >= " + a[1] + ") ? 1.0 : 0.0)";
    case OpType::Less: return "((" + a[0] + " < " + a[1] + ") ? 1.0 : 0.0)";
    case OpType::LessEqual: return "((" + a[0] + " <= " + a[1] + ") ? 1.0 : 0.0)";
    case OpType::LogicalAnd: return "((" + a[0] + " != 0.0 && " + a[1] + " != 0.0) ? 1.0 : 0.0)";
    case OpType::LogicalOr: return "((" + a[0] + " != 0.0 || " + a[1] + " != 0.0) ? 1.0 : 0.0)";
    case OpType::Max: return "max(" + a[0] + ", " + a[1] + ")";
    case OpType::Min: return "min(" + a[0] + ", " + a[1] + ")";
    case OpType::Neg: return "(-" + a[0] + ")";
    case OpType::Abs: return "abs(" + a[0] + ")";
    case OpType::Sqrt: return "sqrt(" + a[0] + ")";
    case OpType::Square: return "(" + a[0] + " * " + a[0] + ")";
    case OpType::Sign: return "sign(" + a[0] + ")";
    case OpType::Floor: return "floor(" + a[0] + ")";
    case OpType::Sin: return "sin(" + a[0] + ")";
    case OpType::Cos: return "cos(" + a[0] + ")";
    case OpType::LogicalNot: return "((" + a[0] + " == 0.0) ? 1.0 : 0.0)";
    case OpType::AsType:
    case OpType::Copy: return a[0];
    case OpType::Where: return "((" + a[0] + " != 0.0) ? " + a[1] + " : " + a[2] + ")";
    case OpType::Clip: return "clamp(" + a[0] + ", " + a[1] + ", " + a[2] + ")";
    default: throw std::runtime_error("mkx fuse: unsupported op in fuse_scalar_expr");
  }
}

struct FusionPlan {
  // unit単位でトポロジカル順に並んだ実行単位。size()==1は非fusionノード、size()>1はfuseされたクラスタ。
  std::vector<std::vector<OpNode*>> units;
  std::unordered_set<OpNode*> materialized;
};

inline FusionPlan build_fusion_plan(const std::vector<NodePtr>& order, const std::unordered_set<OpNode*>& roots) {
  std::unordered_map<OpNode*, size_t> cluster_of;
  std::vector<std::vector<OpNode*>> clusters;

  for(auto& np : order) {
    OpNode* n = np.get();
    if(!fuse_is_fusable(n->type)) continue;
    if(shader_group_for(n->type) == ShaderGroup::Gather) {
      // Gatherは自スレッドのiと異なるindexで入力を読むため入力は常に実バッファである必要がある。吸収せず新規クラスタを開始する。
      clusters.push_back({n});
      cluster_of[n] = clusters.size() - 1;
      continue;
    }
    OpNode* p0 = n->inputs.empty() ? nullptr : n->inputs[0].get();
    auto it = p0 ? cluster_of.find(p0) : cluster_of.end();
    if(p0 && it != cluster_of.end()) {
      clusters[it->second].push_back(n);
      cluster_of[n] = it->second;
    } else {
      clusters.push_back({n});
      cluster_of[n] = clusters.size() - 1;
    }
  }

  // 別クラスタ(または非クラスタ)から読まれるクラスタ化済みノードは実バッファが必要。
  std::unordered_set<OpNode*> materialized;
  for(auto& np : order) {
    OpNode* n = np.get();
    auto nit = cluster_of.find(n);
    for(auto& in : n->inputs) {
      OpNode* b = in.get();
      auto bit = cluster_of.find(b);
      if(bit == cluster_of.end()) continue;
      bool same = (nit != cluster_of.end()) && (nit->second == bit->second);
      if(!same) materialized.insert(b);
    }
  }
  for(auto* r : roots) {
    auto rit = cluster_of.find(r);
    if(rit != cluster_of.end()) materialized.insert(r);
  }

  // unit化: クラスタ(cluster id == unit id、0..clusters.size()-1)+非クラスタノード(新規unit)。
  std::vector<std::vector<OpNode*>> units = clusters;
  std::unordered_map<OpNode*, size_t> unit_of;
  for(auto& [node, cid] : cluster_of) unit_of[node] = cid;
  for(auto& np : order) {
    OpNode* n = np.get();
    if(unit_of.count(n)) continue;
    units.push_back({n});
    unit_of[n] = units.size() - 1;
  }

  size_t N = units.size();
  std::vector<std::unordered_set<size_t>> children(N);
  std::vector<size_t> indegree(N, 0);
  std::vector<size_t> first_seen(N, SIZE_MAX);
  {
    size_t pos = 0;
    for(auto& np : order) {
      size_t u = unit_of.at(np.get());
      if(first_seen[u] == SIZE_MAX) first_seen[u] = pos;
      pos++;
    }
  }
  for(auto& np : order) {
    OpNode* n = np.get();
    size_t cu = unit_of.at(n);
    for(auto& in : n->inputs) {
      OpNode* p = in.get();
      auto pit = unit_of.find(p);
      if(pit == unit_of.end()) continue; // 既に評価済みでtopo_sort対象外(モデル定数等)
      size_t pu = pit->second;
      if(pu == cu) continue;
      if(children[pu].insert(cu).second) indegree[cu]++;
    }
  }

  using Item = std::pair<size_t, size_t>; // (first_seen, unit_id) -- min-heapで決定的な順序にする
  std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
  for(size_t u = 0; u < N; u++) {
    if(indegree[u] == 0) pq.push({first_seen[u], u});
  }
  std::vector<std::vector<OpNode*>> sorted_units;
  sorted_units.reserve(N);
  while(!pq.empty()) {
    auto [_, u] = pq.top();
    pq.pop();
    sorted_units.push_back(units[u]);
    for(size_t c : children[u]) {
      if(--indegree[c] == 0) pq.push({first_seen[c], c});
    }
  }

  return FusionPlan{std::move(sorted_units), std::move(materialized)};
}

template <ComputeBackend Backend>
void eval_fused_cluster(const std::vector<OpNode*>& members, std::unordered_map<size_t, typename Backend::Pipeline>& cache, const std::unordered_set<OpNode*>& materialized) {
  std::unordered_map<OpNode*, size_t> local_idx;
  for(size_t i = 0; i < members.size(); i++) local_idx[members[i]] = i;

  std::vector<OpNode*> ext_inputs;
  std::unordered_map<OpNode*, size_t> ext_idx;
  for(auto* m : members) {
    for(auto& in : m->inputs) {
      OpNode* p = in.get();
      if(local_idx.count(p)) continue;
      if(!ext_idx.count(p)) {
        ext_idx[p] = ext_inputs.size();
        ext_inputs.push_back(p);
      }
    }
  }

  std::vector<OpNode*> outs;
  std::unordered_map<OpNode*, size_t> out_slot;
  for(auto* m : members) {
    if(materialized.count(m)) {
      out_slot[m] = outs.size();
      outs.push_back(m);
    }
  }

  int64_t count = shape_size(members[0]->shape);

  std::string src = "#version 450\nlayout(local_size_x = 256) in;\n";
  for(size_t k = 0; k < ext_inputs.size(); k++) {
    src += "layout(std430, binding = " + std::to_string(k) + ") readonly buffer FIN" + std::to_string(k) + " { float in" + std::to_string(k) + "[]; };\n";
  }
  for(size_t k = 0; k < outs.size(); k++) {
    size_t b = ext_inputs.size() + k;
    src += "layout(std430, binding = " + std::to_string(b) + ") writeonly buffer FOUT" + std::to_string(k) + " { float out" + std::to_string(k) + "[]; };\n";
  }
  src += "void main() {\n  uint i = gl_GlobalInvocationID.x;\n  if (i >= " + std::to_string(count) + "u) return;\n";

  auto ref = [&](OpNode* n) -> std::string {
    auto it = local_idx.find(n);
    if(it != local_idx.end()) return "v" + std::to_string(it->second);
    return "in" + std::to_string(ext_idx.at(n)) + "[i]";
  };

  for(size_t idx = 0; idx < members.size(); idx++) {
    OpNode* n = members[idx];
    std::string vname = "v" + std::to_string(idx);
    if(shader_group_for(n->type) == ShaderGroup::Gather) {
      Push pc = build_push(*n);
      std::string extname = "in" + std::to_string(ext_idx.at(n->inputs[0].get()));
      std::string sfx = std::to_string(idx);
      src += "  float " + vname + ";\n  {\n";
      src += "    uint out_shape_" + sfx + "[4] = uint[4](" + std::to_string(pc.out_shape[0]) + "u," + std::to_string(pc.out_shape[1]) + "u," + std::to_string(pc.out_shape[2]) + "u," + std::to_string(pc.out_shape[3]) + "u);\n";
      src += "    uint in_shape_" + sfx + "[4] = uint[4](" + std::to_string(pc.in_shape[0]) + "u," + std::to_string(pc.in_shape[1]) + "u," + std::to_string(pc.in_shape[2]) + "u," + std::to_string(pc.in_shape[3]) + "u);\n";
      src += "    uint in_strides_" + sfx + "[4] = uint[4](" + std::to_string(pc.in_strides[0]) + "u," + std::to_string(pc.in_strides[1]) + "u," + std::to_string(pc.in_strides[2]) + "u," + std::to_string(pc.in_strides[3]) + "u);\n";
      src += "    uint remaining = i;\n    uint in_index = " + std::to_string(pc.in_base_offset) + "u;\n";
      src += "    for (int d = " + std::to_string(static_cast<int>(pc.ndim) - 1) + "; d >= 0; --d) {\n";
      src += "      uint dim_size = out_shape_" + sfx + "[d];\n      uint comp = remaining % dim_size;\n      remaining /= dim_size;\n";
      src += "      uint src_comp = comp % in_shape_" + sfx + "[d];\n      in_index += src_comp * in_strides_" + sfx + "[d];\n    }\n";
      src += "    " + vname + " = " + extname + "[in_index];\n  }\n";
    } else {
      std::vector<std::string> args;
      args.reserve(n->inputs.size());
      for(auto& in : n->inputs) args.push_back(ref(in.get()));
      src += "  float " + vname + " = " + fuse_scalar_expr(n->type, args) + ";\n";
    }
    auto oit = out_slot.find(n);
    if(oit != out_slot.end()) {
      src += "  out" + std::to_string(oit->second) + "[i] = " + vname + ";\n";
    }
  }
  src += "}\n";

  size_t hash = std::hash<std::string>{}(src);
  auto pit = cache.find(hash);
  if(pit == cache.end()) pit = cache.emplace(hash, Backend::compile(src, hash)).first;

  std::vector<typename Backend::Buffer*> bufs;
  bufs.reserve(ext_inputs.size() + outs.size());
  for(auto* e : ext_inputs) bufs.push_back(static_cast<typename Backend::Buffer*>(e->gpu_buffer));

  std::vector<typename Backend::Buffer*> out_bufs(outs.size());
  for(size_t k = 0; k < outs.size(); k++) {
    out_bufs[k] = Backend::alloc(static_cast<size_t>(count) * dtype_size(outs[k]->dtype));
    bufs.push_back(out_bufs[k]);
  }

  uint32_t groups_x = static_cast<uint32_t>((count + 255) / 256);
  Backend::dispatch(pit->second, bufs, {}, {groups_x, 1, 1});

  for(size_t k = 0; k < outs.size(); k++) {
    OpNode* n = outs[k];
    n->gpu_buffer = out_bufs[k];
    n->evaluated  = true;
    auto* buf     = out_bufs[k];
    n->free_gpu_buffer = [buf]() { Backend::free(buf); };
  }
  for(auto* m : members) {
    if(!materialized.count(m)) {
      m->evaluated  = true;
      m->gpu_buffer = nullptr;
    }
  }
}

} // namespace mkx::detail

namespace mkx {

// mkx::eval<Backend>()相当だが要素単位演算+形状変換をfusionする(他は従来通りeval_node()で個別dispatch)。
template <ComputeBackend Backend> void eval_fused(const std::vector<NodePtr>& roots) {
  std::unordered_set<OpNode*> visited;
  std::vector<NodePtr> order;
  for(auto& r : roots) detail::topo_sort(r, visited, order);

  std::unordered_set<OpNode*> root_set;
  for(auto& r : roots) root_set.insert(r.get());

  auto plan = detail::build_fusion_plan(order, root_set);

  static std::unordered_map<size_t, typename Backend::Pipeline> pipeline_cache;
  for(auto& unit : plan.units) {
    if(unit.size() == 1) {
      detail::eval_node<Backend>(*unit[0], pipeline_cache);
    } else {
      detail::eval_fused_cluster<Backend>(unit, pipeline_cache, plan.materialized);
    }
  }
  Backend::wait_idle();
}

} // namespace mkx
