#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace temporal_dsa::sm89 {

constexpr int kHeads = 64;
constexpr int kHeadDim = 128;
constexpr int kGroupHeads = 8;
constexpr int kGroups = 8;
constexpr int kBlockTokens = 64;
constexpr int kComputeWarps = 4;

__device__ __forceinline__ float negative_infinity() {
  return -__int_as_float(0x7f800000);
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo,
                                               __nv_bfloat16 hi) {
  union Pair {
    __nv_bfloat162 value;
    uint32_t bits;
  } pair;
  pair.value = __halves2bfloat162(lo, hi);
  return pair.bits;
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Each warp owns a 16-token x 8-head output tile. The exact m16n8k16
// instruction avoids padding H8 to 16 columns. Q is packed K-major within a
// group: q_packed[group][k][head]. Only lanes 0,4,...,28 own scalar outputs.
template <int KStride, bool XorSwizzle>
__device__ __forceinline__ int k_smem_index(int row, int col) {
  if constexpr (XorSwizzle) {
    const int vector = (col >> 3) ^ (row & 7);
    return row * KStride + vector * 8 + (col & 7);
  }
  return row * KStride + col;
}

template <int KStride = kHeadDim, bool XorSwizzle = false>
__device__ __forceinline__ void compute_group_layout(
    const __nv_bfloat16* __restrict__ k_smem,
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed, int group, float& row0_score,
    float& row1_score) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int row0 = warp * 16 + row_group;
  const int row1 = row0 + 8;

  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;

#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const int k_base = kt * 16;
    const int a_col0 = k_base + lane_in_group * 2;
    const int a_col1 = k_base + 8 + lane_in_group * 2;
    const uint32_t a0 = pack_bf16(
        k_smem[k_smem_index<KStride, XorSwizzle>(row0, a_col0)],
        k_smem[k_smem_index<KStride, XorSwizzle>(row0, a_col0 + 1)]);
    const uint32_t a1 = pack_bf16(
        k_smem[k_smem_index<KStride, XorSwizzle>(row1, a_col0)],
        k_smem[k_smem_index<KStride, XorSwizzle>(row1, a_col0 + 1)]);
    const uint32_t a2 = pack_bf16(
        k_smem[k_smem_index<KStride, XorSwizzle>(row0, a_col1)],
        k_smem[k_smem_index<KStride, XorSwizzle>(row0, a_col1 + 1)]);
    const uint32_t a3 = pack_bf16(
        k_smem[k_smem_index<KStride, XorSwizzle>(row1, a_col1)],
        k_smem[k_smem_index<KStride, XorSwizzle>(row1, a_col1 + 1)]);

    const int b_row0 = k_base + lane_in_group * 2;
    const int b_row1 = k_base + 8 + lane_in_group * 2;
    const int q_group = group * kHeadDim * kGroupHeads;
    const uint32_t b0 = pack_bf16(
        q_packed[q_group + b_row0 * kGroupHeads + row_group],
        q_packed[q_group + (b_row0 + 1) * kGroupHeads + row_group]);
    const uint32_t b1 = pack_bf16(
        q_packed[q_group + b_row1 * kGroupHeads + row_group],
        q_packed[q_group + (b_row1 + 1) * kGroupHeads + row_group]);
    mma_m16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
  }

  const int head0 = lane_in_group * 2;
  const float w0 = w_packed[group * kGroupHeads + head0];
  const float w1 = w_packed[group * kGroupHeads + head0 + 1];
  float partial0 = w0 * fmaxf(d0, 0.0f) + w1 * fmaxf(d1, 0.0f);
  float partial1 = w0 * fmaxf(d2, 0.0f) + w1 * fmaxf(d3, 0.0f);
  partial0 += __shfl_down_sync(0xffffffffu, partial0, 2, 4);
  partial1 += __shfl_down_sync(0xffffffffu, partial1, 2, 4);
  partial0 += __shfl_down_sync(0xffffffffu, partial0, 1, 4);
  partial1 += __shfl_down_sync(0xffffffffu, partial1, 1, 4);
  if (lane_in_group == 0) {
    row0_score += partial0;
    row1_score += partial1;
  }
}

__device__ __forceinline__ void compute_group(
    const __nv_bfloat16* __restrict__ k_smem,
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed, int group, float& row0_score,
    float& row1_score) {
  compute_group_layout<kHeadDim, false>(k_smem, q_packed, w_packed, group,
                                        row0_score, row1_score);
}

}  // namespace temporal_dsa::sm89
