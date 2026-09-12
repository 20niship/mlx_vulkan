#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mkx/core/backend_concept.hpp>
#include <mkx/core/op_node.hpp>
#include <mkx/shaders/shader_source.hpp>

namespace mkx {

namespace detail {

inline void topo_sort(const NodePtr& node, std::unordered_set<OpNode*>& visited, std::vector<NodePtr>& order) {
  if(!node || node->evaluated || visited.contains(node.get())) return;
  visited.insert(node.get());
  for(auto& in : node->inputs) topo_sort(in, visited, order);
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

template <ComputeBackend Backend> void eval_node(OpNode& node, std::unordered_map<size_t, typename Backend::Pipeline>& cache) {
  if(node.evaluated) return;

  ShaderGroup group = shader_group_for(node.type);

  if(group == ShaderGroup::View) {
    node.gpu_buffer = node.inputs[0]->gpu_buffer;
    node.evaluated  = true;
    return;
  }

  if(node.type == OpType::CustomKernelOutput) {
    auto& owner     = *node.inputs[0];
    node.gpu_buffer = owner.multi_outputs[static_cast<size_t>(node.output_index)];
    node.evaluated  = true;
    return;
  }

  if(node.type == OpType::CustomKernel) {
    std::vector<typename Backend::Buffer*> bufs;
    for(auto& in : node.inputs) {
      bufs.push_back(static_cast<typename Backend::Buffer*>(in->gpu_buffer));
    }
    for(size_t k = 0; k < node.custom_output_shapes.size(); ++k) {
      auto* out = Backend::alloc(static_cast<size_t>(shape_size(node.custom_output_shapes[k])) * dtype_size(node.custom_output_dtypes[k]));
      bufs.push_back(out);
      node.multi_outputs.push_back(out);
    }
    node.free_gpu_buffer = [&node]() {
      for(void* p : node.multi_outputs) Backend::free(static_cast<typename Backend::Buffer*>(p));
    };

    size_t hash = std::hash<std::string>{}(node.custom_source);
    auto it     = cache.find(hash);
    if(it == cache.end()) {
      it = cache.emplace(hash, Backend::compile(node.custom_source, hash)).first;
    }
    Backend::dispatch(it->second, bufs, {}, node.custom_groups);
    node.evaluated = true;
    return;
  }

  if(group == ShaderGroup::CpuFallback) {
    int n        = static_cast<int>(node.shape.back());
    auto* in_buf = static_cast<typename Backend::Buffer*>(node.inputs[0]->gpu_buffer);
    std::vector<float> a(static_cast<size_t>(shape_size(node.inputs[0]->shape)));
    Backend::download(in_buf, a.data(), a.size() * sizeof(float));

    std::vector<float> result;
    if(node.type == OpType::Cholesky) {
      result = cholesky_cpu(a, n);
    } else {
      auto* b_buf = static_cast<typename Backend::Buffer*>(node.inputs[1]->gpu_buffer);
      std::vector<float> b(static_cast<size_t>(n));
      Backend::download(b_buf, b.data(), b.size() * sizeof(float));
      result = forward_substitute(a, b, n);
    }

    auto* out = Backend::alloc(result.size() * sizeof(float));
    Backend::upload(out, result.data(), result.size() * sizeof(float));
    node.gpu_buffer      = out;
    node.free_gpu_buffer = [&node]() { Backend::free(static_cast<typename Backend::Buffer*>(node.gpu_buffer)); };
    node.evaluated       = true;
    return;
  }

  int64_t out_count      = shape_size(node.shape);
  int64_t dispatch_count = (group == ShaderGroup::Reduce) ? shape_size(node.inputs[0]->shape) : out_count;
  auto* out              = Backend::alloc(static_cast<size_t>(out_count) * dtype_size(node.dtype));

  std::string src = shader_source_for(node.type);
  size_t hash     = std::hash<std::string>{}(src);
  auto it         = cache.find(hash);
  if(it == cache.end()) {
    it = cache.emplace(hash, Backend::compile(src, hash)).first;
  }

  std::vector<typename Backend::Buffer*> bufs;
  for(auto& in : node.inputs) {
    bufs.push_back(static_cast<typename Backend::Buffer*>(in->gpu_buffer));
  }
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

  node.gpu_buffer      = out;
  node.free_gpu_buffer = [&node]() { Backend::free(static_cast<typename Backend::Buffer*>(node.gpu_buffer)); };
  node.evaluated       = true;
}

} // namespace detail

// グラフを未evalノードのみトポロジカルソートしてdispatchする。fuse(複数ノード融合)は行わない。
template <ComputeBackend Backend, class... Arrays> void eval(Arrays&... arrs) {
  std::unordered_set<OpNode*> visited;
  std::vector<NodePtr> order;
  (detail::topo_sort(arrs.node(), visited, order), ...);

  static std::unordered_map<size_t, typename Backend::Pipeline> pipeline_cache;
  for(auto& node : order) {
    detail::eval_node<Backend>(*node, pipeline_cache);
  }
  Backend::wait_idle();
}

} // namespace mkx
