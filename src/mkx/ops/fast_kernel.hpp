#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <mkx/core/array.hpp>
#include <mkx/core/op_node.hpp>

namespace mkx::fast {

// mx::fast::metal_kernel互換API。sourceはGLSLのmain()本体のみ(バッファ宣言は自動生成)、次元等はsource自体に文字列展開で焼き込む前提でpush constantは使わない。
struct Kernel {
  std::string name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::string source; // GLSL関数本体(main()の中身相当)
  std::string header; // main()より前に挿入する補助GLSL(関数/定数定義)

  std::string build_full_source(uint32_t local_size_x) const {
    std::string src = "#version 450\n";
    src += "layout(local_size_x = " + std::to_string(local_size_x) + ") in;\n";
    uint32_t binding = 0;
    for(auto& n : input_names) {
      src += "layout(std430, binding = " + std::to_string(binding++) + ") readonly buffer BUF_" + n + " { float " + n + "[]; };\n";
    }
    for(auto& n : output_names) {
      // scratch用途(euler_devmemのL_scratch等)で読み書き両方必要なため書き込み専用にはしない
      src += "layout(std430, binding = " + std::to_string(binding++) + ") buffer BUF_" + n + " { float " + n + "[]; };\n";
    }
    src += header;
    src += "\nvoid main() {\n";
    src += source;
    src += "\n}\n";
    return src;
  }

  std::vector<array<float, 1>> operator()(const std::vector<array<float, 1>>& inputs, const std::vector<Shape>& output_shapes, std::array<uint32_t, 3> grid, std::array<uint32_t, 3> threadgroup, const std::vector<void*>& preallocated_outputs = {}) const {
    auto owner  = std::make_shared<OpNode>();
    owner->type = OpType::CustomKernel;
    for(auto& in : inputs) owner->inputs.push_back(in.node());
    owner->custom_output_shapes  = output_shapes;
    owner->preallocated_outputs  = preallocated_outputs;
    owner->custom_output_dtypes.assign(output_shapes.size(), Dtype::Float32);
    owner->custom_groups = {
      (grid[0] + threadgroup[0] - 1) / threadgroup[0],
      (grid[1] + threadgroup[1] - 1) / threadgroup[1],
      (grid[2] + threadgroup[2] - 1) / threadgroup[2],
    };
    owner->custom_source = build_full_source(threadgroup[0]);

    std::vector<array<float, 1>> outputs;
    for(size_t i = 0; i < output_shapes.size(); ++i) {
      auto alias          = std::make_shared<OpNode>();
      alias->type         = OpType::CustomKernelOutput;
      alias->shape        = output_shapes[i];
      alias->dtype        = Dtype::Float32;
      alias->inputs       = {owner};
      alias->output_index = static_cast<int>(i);
      outputs.emplace_back(alias);
    }
    return outputs;
  }
};

inline Kernel compute_kernel(std::string name, std::vector<std::string> input_names, std::vector<std::string> output_names, std::string source, std::string header = "") { return Kernel{std::move(name), std::move(input_names), std::move(output_names), std::move(source), std::move(header)}; }

} // namespace mkx::fast
