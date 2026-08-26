#pragma once

#include "common_mma.cuh"

namespace temporal_dsa::sm89 {

__device__ __forceinline__ void cp_async_16(void* smem_dst,
                                            const void* global_src,
                                            int valid_bytes) {
  const uint32_t smem_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
                   "r"(smem_addr), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void load_k_sync(
    const __nv_bfloat16* __restrict__ keys, __nv_bfloat16* k_smem,
    int block, int length) {
  constexpr int kVectors = kBlockTokens * kHeadDim / 8;
  const int base_token = block * kBlockTokens;
  for (int vec = threadIdx.x; vec < kVectors; vec += blockDim.x) {
    const int row = vec / (kHeadDim / 8);
    const int col_vec = vec % (kHeadDim / 8);
    uint4 value{};
    if (base_token + row < length) {
      value = reinterpret_cast<const uint4*>(
          keys + (base_token + row) * kHeadDim)[col_vec];
    }
    reinterpret_cast<uint4*>(k_smem)[vec] = value;
  }
}

template <int KStride, bool XorSwizzle>
__device__ __forceinline__ void load_k_sync_layout(
    const __nv_bfloat16* __restrict__ keys, __nv_bfloat16* k_smem,
    int block, int length) {
  constexpr int kVectors = kBlockTokens * kHeadDim / 8;
  const int base_token = block * kBlockTokens;
  for (int vec = threadIdx.x; vec < kVectors; vec += blockDim.x) {
    const int row = vec / (kHeadDim / 8);
    const int col_vec = vec % (kHeadDim / 8);
    uint4 value{};
    if (base_token + row < length) {
      value = reinterpret_cast<const uint4*>(
          keys + (base_token + row) * kHeadDim)[col_vec];
    }
    const int dst_vec = XorSwizzle ? (col_vec ^ (row & 7)) : col_vec;
    reinterpret_cast<uint4*>(k_smem + row * KStride)[dst_vec] = value;
  }
}

__device__ __forceinline__ void prefetch_k_async(
    const __nv_bfloat16* __restrict__ keys, __nv_bfloat16* k_smem,
    int block, int length, int copy_thread, int copy_threads) {
  constexpr int kVectors = kBlockTokens * kHeadDim / 8;
  const int base_token = block * kBlockTokens;
  for (int vec = copy_thread; vec < kVectors; vec += copy_threads) {
    const int row = vec / (kHeadDim / 8);
    const int col_vec = vec % (kHeadDim / 8);
    const bool valid = base_token + row < length;
    const __nv_bfloat16* src = valid
        ? keys + (base_token + row) * kHeadDim + col_vec * 8
        : keys;
    cp_async_16(reinterpret_cast<uint4*>(k_smem) + vec, src,
                valid ? 16 : 0);
  }
  cp_async_commit();
}

}  // namespace temporal_dsa::sm89
