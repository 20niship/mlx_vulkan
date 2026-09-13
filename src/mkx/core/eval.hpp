#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/shaders/shader_source.hpp>
#include <mkx/vulkan/vulkan_backend.hpp>

namespace mkx {

namespace detail {

template <class Backend> void topo_sort(const NodePtr<Backend>& node, std::unordered_set<OpNode<Backend>*>& visited, std::vector<NodePtr<Backend>>& order) {
  if(!node || node->evaluated || visited.contains(node.get())) return;
  visited.insert(node.get());
  for(auto& in : node->inputs) topo_sort<Backend>(in, visited, order);
  order.push_back(node);
}

// n×n下三角行列Lの前進代入でL*x=bを解く(SolveTriangular)。逐次依存が強くGPU向きでないためCPUで解く。
inline std::vector<float> forward_substitute(const std::vector<float>& L, const std::vector<float>& b, int n) {
  std::vector<float> x(static_cast<size_t>(n));
  for(int i = 0; i < n; ++i) {
    float sum = b[static_cast<size_t>(i)];
    for(int j = 0; j < i; ++j) sum -= L[static_cast<size_t>(i * n + j)] * x[static_cast<size_t>(j)];
    x[static_cast<size_t>(i)] = sum / L[static_cast<size_t>(i * n + i)];
  }
  return x;
}

// n×n対称正定値行列Aのコレスキー分解A=L*L^T。逐次依存が強くGPU向きでないためCPUで解く。
inline std::vector<float> cholesky_cpu(const std::vector<float>& a, int n) {
  std::vector<float> L(static_cast<size_t>(n * n), 0.0f);
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j <= i; ++j) {
      float sum = a[static_cast<size_t>(i * n + j)];
      for(int k = 0; k < j; ++k) sum -= L[static_cast<size_t>(i * n + k)] * L[static_cast<size_t>(j * n + k)];
      if(i == j) {
        L[static_cast<size_t>(i * n + j)] = std::sqrt(sum);
      } else {
        L[static_cast<size_t>(i * n + j)] = sum / L[static_cast<size_t>(j * n + j)];
      }
    }
  }
  return L;
}

// ── operator fusion(要素単位演算+形状変換のみ、他はdispatch境界) ──

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

template <class Backend> struct FusionPlan {
  // unit単位でトポロジカル順に並んだ実行単位。size()==1は非fusionノード、size()>1はfuseされたクラスタ。
  std::vector<std::vector<NodePtr<Backend>>> units;
  std::unordered_set<OpNode<Backend>*> materialized;
};

template <class Backend> FusionPlan<Backend> build_fusion_plan(const std::vector<NodePtr<Backend>>& order, const std::unordered_set<OpNode<Backend>*>& roots) {
  std::unordered_map<OpNode<Backend>*, size_t> cluster_of;
  std::vector<std::vector<NodePtr<Backend>>> clusters;

  for(auto& np : order) {
    OpNode<Backend>* n = np.get();
    if(!fuse_is_fusable(n->type)) continue;
    if(shader_group_for(n->type) == ShaderGroup::Gather) {
      // Gatherは自スレッドのiと異なるindexで入力を読むため入力は常に実バッファである必要がある。吸収せず新規クラスタを開始する。
      clusters.push_back({np});
      cluster_of[n] = clusters.size() - 1;
      continue;
    }
    OpNode<Backend>* p0 = n->inputs.empty() ? nullptr : n->inputs[0].get();
    auto it              = p0 ? cluster_of.find(p0) : cluster_of.end();
    if(p0 && it != cluster_of.end()) {
      clusters[it->second].push_back(np);
      cluster_of[n] = it->second;
    } else {
      clusters.push_back({np});
      cluster_of[n] = clusters.size() - 1;
    }
  }

  // 別クラスタ(または非クラスタ)から読まれるクラスタ化済みノードは実バッファが必要。
  std::unordered_set<OpNode<Backend>*> materialized;
  for(auto& np : order) {
    OpNode<Backend>* n = np.get();
    auto nit            = cluster_of.find(n);
    for(auto& in : n->inputs) {
      OpNode<Backend>* b = in.get();
      auto bit             = cluster_of.find(b);
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
  std::vector<std::vector<NodePtr<Backend>>> units = clusters;
  std::unordered_map<OpNode<Backend>*, size_t> unit_of;
  for(auto& [node, cid] : cluster_of) unit_of[node] = cid;
  for(auto& np : order) {
    OpNode<Backend>* n = np.get();
    if(unit_of.count(n)) continue;
    units.push_back({np});
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
    OpNode<Backend>* n = np.get();
    size_t cu           = unit_of.at(n);
    for(auto& in : n->inputs) {
      OpNode<Backend>* p = in.get();
      auto pit             = unit_of.find(p);
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
  std::vector<std::vector<NodePtr<Backend>>> sorted_units;
  sorted_units.reserve(N);
  while(!pq.empty()) {
    auto [_, u] = pq.top();
    pq.pop();
    sorted_units.push_back(units[u]);
    for(size_t c : children[u]) {
      if(--indegree[c] == 0) pq.push({first_seen[c], c});
    }
  }

  // 貪欲クラスタリングが2クラスタ間に循環依存を作ることがあり、その場合Kahn's sortが全unitを消費できない。正しさ優先でfusionなしにフォールバックする。
  if(sorted_units.size() != N) {
    std::vector<std::vector<NodePtr<Backend>>> fallback;
    fallback.reserve(order.size());
    for(auto& np : order) fallback.push_back({np});
    return FusionPlan<Backend>{std::move(fallback), {}};
  }

  return FusionPlan<Backend>{std::move(sorted_units), std::move(materialized)};
}

template <class Backend>
void eval_fused_cluster(const std::vector<NodePtr<Backend>>& members, std::unordered_map<size_t, typename Backend::Pipeline>& cache, const std::unordered_set<OpNode<Backend>*>& materialized) {
  std::unordered_map<OpNode<Backend>*, size_t> local_idx;
  for(size_t i = 0; i < members.size(); i++) local_idx[members[i].get()] = i;

  std::vector<NodePtr<Backend>> ext_inputs;
  std::unordered_map<OpNode<Backend>*, size_t> ext_idx;
  for(auto& m : members) {
    for(auto& in : m->inputs) {
      OpNode<Backend>* p = in.get();
      if(local_idx.count(p)) continue;
      if(!ext_idx.count(p)) {
        ext_idx[p] = ext_inputs.size();
        ext_inputs.push_back(in);
      }
    }
  }

  std::vector<NodePtr<Backend>> outs;
  std::unordered_map<OpNode<Backend>*, size_t> out_slot;
  for(auto& m : members) {
    if(materialized.count(m.get())) {
      out_slot[m.get()] = outs.size();
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

  auto ref = [&](OpNode<Backend>* n) -> std::string {
    auto it = local_idx.find(n);
    if(it != local_idx.end()) return "v" + std::to_string(it->second);
    return "in" + std::to_string(ext_idx.at(n)) + "[i]";
  };

  for(size_t idx = 0; idx < members.size(); idx++) {
    OpNode<Backend>* n = members[idx].get();
    std::string vname    = "v" + std::to_string(idx);
    if(shader_group_for(n->type) == ShaderGroup::Gather) {
      Push pc              = build_push(*n);
      std::string extname = "in" + std::to_string(ext_idx.at(n->inputs[0].get()));
      std::string sfx      = std::to_string(idx);
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
  auto pit     = cache.find(hash);
  if(pit == cache.end()) pit = cache.emplace(hash, Backend::compile(src, hash)).first;

  std::vector<typename Backend::Buffer*> bufs;
  bufs.reserve(ext_inputs.size() + outs.size());
  for(auto& e : ext_inputs) bufs.push_back(buffer_for<Backend>(e));

  std::vector<typename Backend::Buffer*> out_bufs(outs.size());
  for(size_t k = 0; k < outs.size(); k++) {
    out_bufs[k] = Backend::get_or_allocate(outs[k].get(), static_cast<size_t>(count) * dtype_size(outs[k]->dtype));
    bufs.push_back(out_bufs[k]);
  }

  uint32_t groups_x = static_cast<uint32_t>((count + 255) / 256);
  Backend::dispatch(pit->second, bufs, {}, {groups_x, 1, 1});

  for(auto& m : members) m->evaluated = true;
}

template <class Backend> void eval_node(OpNode<Backend>& node, std::unordered_map<size_t, typename Backend::Pipeline>& cache) {
  if(node.evaluated) return;

  if(node.type == OpType::Const) {
    // Vulkan/MoltenVKは0バイトのvkAllocateMemoryを拒否するため、shapeに0次元を含む(要素数0の)Constでも最低1要素分は確保する。
    size_t nbytes = std::max<size_t>(static_cast<size_t>(shape_size(node.shape)), 1) * dtype_size(node.dtype);
    auto* buf     = Backend::get_or_allocate(&node, nbytes);
    if(!node.host_data.empty()) Backend::upload(buf, node.host_data.data(), node.host_data.size());
    node.evaluated = true;
    return;
  }

  ShaderGroup group = shader_group_for(node.type);

  if(group == ShaderGroup::View) {
    node.evaluated = true; // バッファはbuffer_for()が入力側へ遡って解決する、View自身は何も確保しない
    return;
  }

  if(node.type == OpType::CustomKernelOutput) {
    node.evaluated = true; // トポロジカル順により所有者は既にdispatch済み、バッファはoutput_aliases経由で確保されている
    return;
  }

  if(node.type == OpType::CustomKernel) {
    std::vector<typename Backend::Buffer*> bufs;
    for(auto& in : node.inputs) bufs.push_back(buffer_for<Backend>(in));

    std::vector<typename Backend::Buffer*> fallback_bufs; // 別名ノードが既に破棄されている場合の一時バッファ(dispatch後すぐ解放)
    for(size_t k = 0; k < node.custom_output_shapes.size(); ++k) {
      size_t nbytes         = static_cast<size_t>(shape_size(node.custom_output_shapes[k])) * dtype_size(node.custom_output_dtypes[k]);
      auto alias             = (k < node.output_aliases.size()) ? node.output_aliases[k].lock() : nullptr;
      typename Backend::Buffer* out;
      if(alias) {
        out             = Backend::get_or_allocate(alias.get(), nbytes);
        alias->evaluated = true;
      } else {
        out = Backend::alloc(nbytes);
        fallback_bufs.push_back(out);
      }
      bufs.push_back(out);
    }

    size_t hash = std::hash<std::string>{}(node.custom_source);
    auto it     = cache.find(hash);
    if(it == cache.end()) {
      it = cache.emplace(hash, Backend::compile(node.custom_source, hash)).first;
    }
    Backend::dispatch(it->second, bufs, {}, node.custom_groups);
    for(auto* fb : fallback_bufs) Backend::free(fb);
    node.evaluated = true;
    return;
  }

  if(group == ShaderGroup::CpuFallback) {
    int n        = static_cast<int>(node.shape.back());
    auto* in_buf = buffer_for<Backend>(node.inputs[0]);
    std::vector<float> a(static_cast<size_t>(shape_size(node.inputs[0]->shape)));
    Backend::download(in_buf, a.data(), a.size() * sizeof(float));

    std::vector<float> result;
    if(node.type == OpType::Cholesky) {
      result = cholesky_cpu(a, n);
    } else {
      auto* b_buf = buffer_for<Backend>(node.inputs[1]);
      std::vector<float> b(static_cast<size_t>(n));
      Backend::download(b_buf, b.data(), b.size() * sizeof(float));
      result = forward_substitute(a, b, n);
    }

    auto* out = Backend::get_or_allocate(&node, result.size() * sizeof(float));
    Backend::upload(out, result.data(), result.size() * sizeof(float));
    node.evaluated = true;
    return;
  }

  int64_t out_count      = shape_size(node.shape);
  int64_t dispatch_count = (group == ShaderGroup::Reduce) ? shape_size(node.inputs[0]->shape) : out_count;
  auto* out              = Backend::get_or_allocate(&node, static_cast<size_t>(out_count) * dtype_size(node.dtype));

  std::string src = shader_source_for(node.type);
  size_t hash     = std::hash<std::string>{}(src);
  auto it         = cache.find(hash);
  if(it == cache.end()) {
    it = cache.emplace(hash, Backend::compile(src, hash)).first;
  }

  std::vector<typename Backend::Buffer*> bufs;
  for(auto& in : node.inputs) bufs.push_back(buffer_for<Backend>(in));
  bufs.push_back(out);

  Push push                         = build_push(node);
  push.count                        = static_cast<uint32_t>(dispatch_count);
  std::vector<std::byte> push_bytes = pack_push(push);

  uint32_t groups_x;
  if(group == ShaderGroup::Reduce) {
    groups_x = 1u; // 単一work-groupが入力全体をgrid-strideで処理する
  } else if(group == ShaderGroup::ReduceAxis) {
    groups_x = static_cast<uint32_t>(out_count); // 出力要素1個=work-group1個
  } else {
    groups_x = static_cast<uint32_t>((dispatch_count + 255) / 256);
  }

  Backend::dispatch(it->second, bufs, push_bytes, {groups_x, 1, 1});
  node.evaluated = true;
}

} // namespace detail

// グラフを未evalノードのみトポロジカルソートし、要素単位演算+形状変換をfusionしてdispatchする。
template <ComputeBackend Backend = VulkanBackend, class... Arrays> void eval(Arrays&... arrs) {
  std::unordered_set<OpNode<Backend>*> visited;
  std::vector<NodePtr<Backend>> order;
  (detail::topo_sort<Backend>(arrs.node(), visited, order), ...);

  std::unordered_set<OpNode<Backend>*> roots = {arrs.node().get()...};

  static std::unordered_map<size_t, typename Backend::Pipeline> pipeline_cache;
  auto plan = detail::build_fusion_plan<Backend>(order, roots);
  for(auto& unit : plan.units) {
    if(unit.size() == 1) {
      detail::eval_node<Backend>(*unit[0], pipeline_cache);
    } else {
      detail::eval_fused_cluster<Backend>(unit, pipeline_cache, plan.materialized);
    }
  }
  Backend::wait_idle();
}

// 動的な出力配列(mx::compile()の戻り値等)を評価するためのNodePtr vector版。eval(Arrays&...)とのオーバーロード曖昧さを避けるため別名にする。
template <ComputeBackend Backend = VulkanBackend> void eval_nodes(const std::vector<NodePtr<Backend>>& roots_nodes) {
  std::unordered_set<OpNode<Backend>*> visited;
  std::vector<NodePtr<Backend>> order;
  for(auto& r : roots_nodes) detail::topo_sort<Backend>(r, visited, order);

  std::unordered_set<OpNode<Backend>*> roots;
  for(auto& r : roots_nodes) roots.insert(r.get());

  static std::unordered_map<size_t, typename Backend::Pipeline> pipeline_cache;
  auto plan = detail::build_fusion_plan<Backend>(order, roots);
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
