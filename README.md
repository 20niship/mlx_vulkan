# mkx — Vulkanで動くMLX互換のGPU計算ライブラリ

MLX(Apple)のAPIにできるだけ近い形で、Vulkan(将来的にOpenCL/CUDAも)上で動くC++20ライブラリ。
計算は遅延評価グラフとして構築され、`mkx::eval<Backend>()`を呼んだ時点でGLSLシェーダを生成・
コンパイル・実行する。backendはテンプレート引数として渡し、コンパイル時に静的に確定する
(仮想関数を使わない)。

現状の実装状況・既知の制約はすべて[todo.md](./todo.md)にまとめてある。MuJoCo-MLX-Cpp
(https://github.com/arghyasur1991/MuJoCo-MLX-Cpp) をこのライブラリ上で動かすことを目標に、
MLX互換の演算(要素演算・形状操作・線形代数・乱数・関数変換)を実装している。

## 必要環境

- C++20対応コンパイラ(AppleClang / GCC / Clang)
- CMake 3.20以上
- Vulkan SDK(ヘッダ・ローダー・`glslangValidator`)
  - macOS: Vulkan SDK(MoltenVK込み)をインストール
  - Linux: `libvulkan-dev` `glslang-tools` + 実行環境にVulkan ICD(実GPU、または
    `mesa-vulkan-drivers`のlavapipeでソフトウェア実行も可)

## ビルド・テスト

```sh
cmake -B build
cmake --build build -j8
./build/mkx_tests
```

`mkx_tests`はdoctest(CMake FetchContentで自動取得)ベースのテストスイートで、実際に
Vulkanデバイスへdispatchして結果を検証する(モックなし)。CI(`.github/workflows/ci.yml`)は
Ubuntu + lavapipe(ソフトウェアVulkan)上で同じテストを実行する。

## 使い方

### 配列生成・要素演算

```cpp
#include "core/array.hpp"
#include "core/eval.hpp"
#include "ops/creation.hpp"
#include "ops/elementwise.hpp"
#include "vulkan/vulkan_backend.hpp"

using mkx::VulkanBackend;

auto a = mkx::zeros<float, 1>({4});
auto b = mkx::ones<float, 1>({4});

// mkx::addを呼んだ時点では計算しない。グラフにノードを積むだけ。
auto c = mkx::add(a, b);

// eval()で初めてshader生成→コンパイル→実機dispatchが走る。
mkx::eval<VulkanBackend>(c);

std::vector<float> result = c.to_vector<VulkanBackend>(); // [1,1,1,1]
```

要素演算は`mkx/src/ops/elementwise.hpp`に一通り揃っている:
`add/subtract/multiply/divide/power/maximum/minimum/equal/greater/greater_equal/less/
less_equal/logical_and/logical_or/negative/abs/sqrt/square/sign/floor/sin/cos/
logical_not/where/clip`。

### 形状操作

```cpp
#include "ops/shape.hpp"

auto m = mkx::reshape<float, 1, 2>(flat, mkx::Shape{3, 4}); // ndim変更はテンプレート引数で明示
auto t = mkx::transpose(m, {1, 0});
auto s = mkx::slice(m, /*starts=*/{0, 1}, /*stops=*/{3, 3});
auto cat = mkx::concatenate(a, b, /*axis=*/0);
```

`reshape/flatten/transpose/broadcast_to/tile/slice/concatenate/stack/take/diag/tril/
triu/copy`に対応。`reshape/flatten`は純粋なview(GPUバッファのコピーなし)。

### 線形代数・リダクション

```cpp
#include "ops/linalg.hpp"

auto s = mkx::sum(a);                 // 全体リダクション
auto row_sums = mkx::sum_axis(m, 1);  // 軸指定リダクション(work-group並列)
auto c = mkx::matmul(a, b);
auto l = mkx::cholesky(m);            // CPUフォールバック(小行列はGPU向きでないため)
auto x = mkx::solve_triangular(l, rhs);
```

`sum/reduce_max/argmax/argmin`(全体)、`sum_axis/reduce_max_axis`(軸指定)、`matmul`、
`cholesky/solve_triangular`(CPUフォールバック)、`cross`に対応。

### 乱数

```cpp
#include "ops/random.hpp"

auto k = mkx::random::key(42);
auto r = mkx::random::normal<float, 1>(k, {1000}); // 標準正規分布
```

### 関数変換(vmap/compile)

```cpp
#include "ops/transforms.hpp"

// shape非依存な関数(要素演算のみ)ならbatch軸込みで1回呼ぶだけで済む(1 dispatch、host loop無し)。
auto generic_fn = []<class T, size_t N>(const mkx::array<T, N>& x) { return mkx::square(x); };
auto vfn = mkx::vmap<float, 2>(generic_fn, /*in_axis=*/0, /*out_axis=*/0);
auto out = vfn(batched_input); // shape (batch, dof) -> (batch, dof)

// compile()は現状passthrough(eval()自体がshaderをハッシュキャッシュしているため)。
auto compiled = mkx::compile(some_fn);
```

### カスタムカーネル(mx.fast.metal_kernel互換)

```cpp
#include "ops/fast_kernel.hpp"

auto kernel = mkx::fast::compute_kernel(
    "my_kernel",
    /*input_names=*/{"a", "b"},
    /*output_names=*/{"sum_out"},
    /*source=*/"sum_out[gl_GlobalInvocationID.x] = a[gl_GlobalInvocationID.x] + b[gl_GlobalInvocationID.x];");

auto outputs = kernel({a, b}, /*output_shapes=*/{mkx::Shape{4}}, /*grid=*/{4,1,1}, /*threadgroup=*/{4,1,1});
```

`source`はGLSLの`main()`本体のみ(バッファ宣言・`main()`は自動生成)。次元等のモデル依存
パラメータは文字列展開でsourceに直接焼き込む(MLXの`metal_kernel`がモデルごとにソースを
再生成する設計を踏襲)。実例は`mkx/tests/mujoco/`にMuJoCoの6カーネル(kinematics/euler/
euler_devmem/forward/collision/solver)の移植がある。

## 設計メモ

- backendは`ComputeBackend` concept(`mkx/src/core/backend_concept.hpp`)を満たす型として
  テンプレート引数で渡す。現状`VulkanBackend`のみ実装済み(OpenCL/CUDAは未着手)。
- 全pipelineは共通の68byte push constant構造体を共有し、演算グループ(`ShaderGroup`)ごとに
  フィールドの意味を使い回す設計(詳細は`mkx/src/shaders/shader_source.hpp`のコメント参照)。
- ライブラリ本体(`mkx/src`)はMLX互換レイヤーの提供に徹し、MuJoCo固有のカーネル実装は
  `mkx/tests/mujoco/`(利用例・テストの位置づけ)に置いている。

## ディレクトリ構成

```
src/
  mkx/
    core/     配列・グラフノード・backend concept・eval
    ops/      要素演算・形状操作・線形代数・乱数・関数変換・カスタムカーネルAPI
    shaders/  GLSLテンプレート・push constant組み立て
    vulkan/   Vulkan backend実装
tests/      doctestベースのテスト(mujoco/配下にMuJoCoカーネル移植例)
```