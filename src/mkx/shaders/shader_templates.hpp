#pragma once

#include <string_view>

namespace mkx::shaders {

// 全pipeline共通の固定push constant layout(VulkanBackend側が単一サイズしか持たないため)。フィールドはop群ごとに意味を使い回す(詳細はshader_source.hppのbuild_push参照)。
inline constexpr std::string_view push_decl = R"GLSL(
layout(push_constant) uniform Push {
    uint count;
    float p0;
    float p1;
    uint ndim;
    uint out_shape[4];
    uint in_shape[4];
    uint in_strides[4];
    uint in_base_offset;
} pc;
)GLSL";

inline constexpr std::string_view creation_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer OUT { SCALAR o[]; };

uint mkx_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
#if OPCODE == 0
    o[i] = SCALAR(pc.p0);
#elif OPCODE == 1
    o[i] = SCALAR(pc.p0 + float(i) * pc.p1);
#elif OPCODE == 2
    uint n = uint(pc.p0);
    uint r = i / n;
    uint c = i % n;
    o[i] = (r == c) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 3
    uint seed = floatBitsToUint(pc.p0);
    uint h1 = mkx_hash_u32(seed ^ (i * 2u));
    uint h2 = mkx_hash_u32(seed ^ (i * 2u + 1u));
    float u1 = (float(h1) + 1.0) / 4294967296.0;
    float u2 = float(h2) / 4294967296.0;
    o[i] = SCALAR(sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2));
#endif
}
)GLSL";

inline constexpr std::string_view unary_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    float v = float(a[i]);
#if OPCODE == 0
    o[i] = SCALAR(-v);
#elif OPCODE == 1
    o[i] = SCALAR(abs(v));
#elif OPCODE == 2
    o[i] = SCALAR(sqrt(v));
#elif OPCODE == 3
    o[i] = SCALAR(v * v);
#elif OPCODE == 4
    o[i] = SCALAR(sign(v));
#elif OPCODE == 5
    o[i] = SCALAR(floor(v));
#elif OPCODE == 6
    o[i] = SCALAR(sin(v));
#elif OPCODE == 7
    o[i] = SCALAR(cos(v));
#elif OPCODE == 8
    o[i] = (v == 0.0) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 9
    o[i] = SCALAR(v);
#elif OPCODE == 10
    { uint n = uint(pc.p0); uint r = i / n; uint c = i % n; o[i] = (r == c) ? SCALAR(float(a[r])) : SCALAR(0.0); }
#elif OPCODE == 11
    { uint n = uint(pc.p0); uint r = i / n; uint c = i % n; o[i] = (c <= r) ? SCALAR(v) : SCALAR(0.0); }
#elif OPCODE == 12
    { uint n = uint(pc.p0); uint r = i / n; uint c = i % n; o[i] = (c >= r) ? SCALAR(v) : SCALAR(0.0); }
#endif
}
)GLSL";

inline constexpr std::string_view binary_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR b[]; };
layout(std430, binding = 2) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    float av = float(a[i]);
    float bv = float(b[i]);
#if OPCODE == 0
    o[i] = SCALAR(av + bv);
#elif OPCODE == 1
    o[i] = SCALAR(av - bv);
#elif OPCODE == 2
    o[i] = SCALAR(av * bv);
#elif OPCODE == 3
    o[i] = SCALAR(av / bv);
#elif OPCODE == 4
    o[i] = SCALAR(pow(av, bv));
#elif OPCODE == 5
    o[i] = (av == bv) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 6
    o[i] = (av > bv) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 7
    o[i] = (av >= bv) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 8
    o[i] = (av < bv) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 9
    o[i] = (av <= bv) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 10
    o[i] = (av != 0.0 && bv != 0.0) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 11
    o[i] = (av != 0.0 || bv != 0.0) ? SCALAR(1.0) : SCALAR(0.0);
#elif OPCODE == 12
    o[i] = SCALAR(max(av, bv));
#elif OPCODE == 13
    o[i] = SCALAR(min(av, bv));
#elif OPCODE == 14
    {
        uint base = (i / 3u) * 3u;
        uint c = i - base;
        uint c1 = (c + 1u) % 3u;
        uint c2 = (c + 2u) % 3u;
        o[i] = SCALAR(float(a[base + c1]) * float(b[base + c2]) - float(a[base + c2]) * float(b[base + c1]));
    }
#endif
}
)GLSL";

inline constexpr std::string_view ternary_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR b[]; };
layout(std430, binding = 2) readonly buffer C { SCALAR c[]; };
layout(std430, binding = 3) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
#if OPCODE == 0
    o[i] = (float(a[i]) != 0.0) ? b[i] : c[i];
#elif OPCODE == 1
    o[i] = SCALAR(clamp(float(a[i]), float(b[i]), float(c[i])));
#endif
}
)GLSL";

// Transpose/Slice/Tile/BroadcastTo共通: in_strides=0でbroadcast、in_shapeでtileのmodulo、in_base_offsetでslice先頭をそれぞれ表現するgather。
inline constexpr std::string_view gather_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    uint remaining = i;
    uint in_index = pc.in_base_offset;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.out_shape[d];
        uint comp = remaining % dim_size;
        remaining /= dim_size;
        uint src_comp = comp % pc.in_shape[d];
        in_index += src_comp * pc.in_strides[d];
    }
    o[i] = a[in_index];
}
)GLSL";

// Concatenate/Stack: axis(in_base_offset)方向のindexがsplit(p0)未満ならA、以上ならBを読む(a/b_stridesはin_shape/in_stridesを流用)。
inline constexpr std::string_view concat_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR b[]; };
layout(std430, binding = 2) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    uint axis = pc.in_base_offset;
    uint split = uint(pc.p0);

    uint idx[4];
    uint remaining = i;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        idx[d] = remaining % pc.out_shape[d];
        remaining /= pc.out_shape[d];
    }

    if (idx[axis] < split) {
        uint lin = 0;
        for (int d = 0; d < int(pc.ndim); ++d) lin += idx[d] * pc.in_shape[d];
        o[i] = a[lin];
    } else {
        idx[axis] -= split;
        uint lin = 0;
        for (int d = 0; d < int(pc.ndim); ++d) lin += idx[d] * pc.in_strides[d];
        o[i] = b[lin];
    }
}
)GLSL";

// Take: axis(in_base_offsetに格納)方向をindices bufferの値で置き換えてdataからgatherする。
inline constexpr std::string_view take_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) readonly buffer IDX { SCALAR idxbuf[]; };
layout(std430, binding = 2) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;

    uint axis = pc.in_base_offset;
    uint idx[4];
    uint remaining = i;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        idx[d] = remaining % pc.out_shape[d];
        remaining /= pc.out_shape[d];
    }

    uint gathered = uint(round(float(idxbuf[idx[axis]])));
    uint lin = 0;
    for (int d = 0; d < int(pc.ndim); ++d) {
        uint comp = (uint(d) == axis) ? gathered : idx[d];
        lin += comp * pc.in_strides[d];
    }
    o[i] = a[lin];
}
)GLSL";

// ponytail: 単一work-group(256スレッド)のgrid-stride全体リダクション。軸指定はreduce_axis_glsl参照。shared中間値は精度維持のため常にfloatで持つ。
inline constexpr std::string_view reduce_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) writeonly buffer OUT { SCALAR o[]; };
shared float sdata[256];
shared uint sidx[256];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint n = pc.count;
#if OPCODE == 0
    float acc = 0.0;
    for (uint i = tid; i < n; i += 256u) acc += float(a[i]);
    sdata[tid] = acc;
#elif OPCODE == 1
    float acc = -3.402823e38;
    for (uint i = tid; i < n; i += 256u) acc = max(acc, float(a[i]));
    sdata[tid] = acc;
#else
    float best = (OPCODE == 2) ? -3.402823e38 : 3.402823e38;
    uint besti = 0u;
    for (uint i = tid; i < n; i += 256u) {
        float v = float(a[i]);
        if ((OPCODE == 2 && v > best) || (OPCODE == 3 && v < best)) { best = v; besti = i; }
    }
    sdata[tid] = best;
    sidx[tid] = besti;
#endif
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
#if OPCODE == 0
            sdata[tid] += sdata[tid + s];
#elif OPCODE == 1
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
#else
            bool better = (OPCODE == 2) ? (sdata[tid + s] > sdata[tid]) : (sdata[tid + s] < sdata[tid]);
            if (better) { sdata[tid] = sdata[tid + s]; sidx[tid] = sidx[tid + s]; }
#endif
        }
        barrier();
    }
    if (tid == 0u) {
#if OPCODE == 2 || OPCODE == 3
        o[0] = SCALAR(float(sidx[0]));
#else
        o[0] = SCALAR(sdata[0]);
#endif
    }
}
)GLSL";

// ponytail: タイル化なし素朴GEMM(M,Nはout_shape、Kはin_base_offset、ndim==3ならout_shape=[B,M,N]のバッチGEMM)。累積はfloatで行い最後にSCALARへ丸める。
inline constexpr std::string_view matmul_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) readonly buffer B { SCALAR b[]; };
layout(std430, binding = 2) writeonly buffer OUT { SCALAR o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) return;
    uint K = pc.in_base_offset;
    if (pc.ndim == 3u) {
        uint M = pc.out_shape[1];
        uint N = pc.out_shape[2];
        uint batch = i / (M * N);
        uint rem   = i % (M * N);
        uint r = rem / N;
        uint c = rem % N;
        float acc = 0.0;
        uint a_base = batch * M * K;
        uint b_base = batch * K * N;
        for (uint k = 0u; k < K; ++k) acc += float(a[a_base + r * K + k]) * float(b[b_base + k * N + c]);
        o[i] = SCALAR(acc);
        return;
    }
    uint N = pc.out_shape[1];
    uint r = i / N;
    uint c = i % N;
    float acc = 0.0;
    for (uint k = 0u; k < K; ++k) acc += float(a[r * K + k]) * float(b[k * N + c]);
    o[i] = SCALAR(acc);
}
)GLSL";

// n×n対称正定値行列Aのコレスキー分解A=L*L^T。逐次依存が強いためバッチ要素(pc.count個)1個=1スレッドで逐次計算する(タイル並列化はせず、Bが並列度を提供する)。
inline constexpr std::string_view cholesky_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) buffer OUT { SCALAR o[]; };
void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= pc.count) return;
    uint n = pc.in_base_offset;
    uint base = e * n * n;
    for (uint i = 0u; i < n; ++i) {
        for (uint j = 0u; j < n; ++j) {
            if (j > i) { o[base + i * n + j] = SCALAR(0.0); continue; }
            float sum = float(a[base + i * n + j]);
            for (uint k = 0u; k < j; ++k) sum -= float(o[base + i * n + k]) * float(o[base + j * n + k]);
            if (i == j) {
                o[base + i * n + j] = SCALAR(sqrt(sum));
            } else {
                o[base + i * n + j] = SCALAR(sum / float(o[base + j * n + j]));
            }
        }
    }
}
)GLSL";

// 下三角行列L(binding0)によるL*x=b(binding1)の前進代入。バッチ要素1個=1スレッド。
inline constexpr std::string_view solve_triangular_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer L { SCALAR l[]; };
layout(std430, binding = 1) readonly buffer BVEC { SCALAR bv[]; };
layout(std430, binding = 2) buffer OUT { SCALAR o[]; };
void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= pc.count) return;
    uint n = pc.in_base_offset;
    uint lbase = e * n * n;
    uint vbase = e * n;
    for (uint i = 0u; i < n; ++i) {
        float sum = float(bv[vbase + i]);
        for (uint j = 0u; j < i; ++j) sum -= float(l[lbase + i * n + j]) * float(o[vbase + j]);
        o[vbase + i] = SCALAR(sum / float(l[lbase + i * n + i]));
    }
}
)GLSL";

// 軸指定リダクション: 出力要素1個=work-group1個を割り当て、axis(in_base_offset)方向をgrid-strideでtree reduce(最大4次元)。shared中間値は精度維持のため常にfloat。
inline constexpr std::string_view reduce_axis_glsl = R"GLSL(
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { SCALAR a[]; };
layout(std430, binding = 1) writeonly buffer OUT { SCALAR o[]; };
shared float sdata[256];
void main() {
    uint out_idx = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint axis = pc.in_base_offset;
    uint axis_len = pc.in_shape[axis];
    uint out_ndim = pc.ndim - 1u;

    uint idx[4];
    uint remaining = out_idx;
    for (int d = int(out_ndim) - 1; d >= 0; --d) {
        idx[d] = remaining % pc.out_shape[d];
        remaining /= pc.out_shape[d];
    }

    uint base = 0u;
    uint in_d = 0u;
    for (uint d = 0u; d < out_ndim; ++d) {
        if (in_d == axis) in_d++;
        base += idx[d] * pc.in_strides[in_d];
        in_d++;
    }

#if OPCODE == 0
    float acc = 0.0;
    for (uint i = tid; i < axis_len; i += 256u) acc += float(a[base + i * pc.in_strides[axis]]);
#else
    float acc = -3.402823e38;
    for (uint i = tid; i < axis_len; i += 256u) acc = max(acc, float(a[base + i * pc.in_strides[axis]]));
#endif
    sdata[tid] = acc;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
#if OPCODE == 0
            sdata[tid] += sdata[tid + s];
#else
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
#endif
        }
        barrier();
    }
    if (tid == 0u) o[out_idx] = SCALAR(sdata[0]);
}
)GLSL";

} // namespace mkx::shaders
