#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "async_copy.cuh"
#include "common_mma.cuh"

namespace td = temporal_dsa::sm89;

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be BF16")
#define CHECK_F32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be FP32")
#define CHECK_I32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

inline void check_qwk(const torch::Tensor& q_packed,
                      const torch::Tensor& w_packed,
                      const torch::Tensor& keys) {
  CHECK_CUDA(q_packed);
  CHECK_CUDA(w_packed);
  CHECK_CUDA(keys);
  CHECK_CONTIGUOUS(q_packed);
  CHECK_CONTIGUOUS(w_packed);
  CHECK_CONTIGUOUS(keys);
  CHECK_BF16(q_packed);
  CHECK_F32(w_packed);
  CHECK_BF16(keys);
  TORCH_CHECK(q_packed.numel() == td::kHeads * td::kHeadDim,
              "q_packed must contain 64x128 elements");
  TORCH_CHECK(w_packed.numel() == td::kHeads,
              "w_packed must contain 64 elements");
  TORCH_CHECK(keys.dim() == 2 && keys.size(1) == td::kHeadDim,
              "keys must be [L,128]");
}

inline cudaStream_t current_stream(const torch::Tensor& tensor) {
  c10::cuda::CUDAGuard guard(tensor.device());
  return at::cuda::getCurrentCUDAStream(tensor.get_device());
}

__global__ void pack_qw_kernel(
    const __nv_bfloat16* __restrict__ q, const float* __restrict__ w,
    const int* __restrict__ fixed_heads, bool use_fixed,
    __nv_bfloat16* __restrict__ q_packed, float* __restrict__ w_packed,
    int* __restrict__ packed_ids) {
  __shared__ int order[td::kHeads];
  __shared__ int used[td::kHeads];
  if (threadIdx.x < td::kHeads) {
    used[threadIdx.x] = 0;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (use_fixed) {
      for (int slot = 0; slot < td::kGroupHeads; ++slot) {
        const int head = fixed_heads[slot];
        order[slot] = head;
        used[head] = 1;
      }
    } else {
      for (int slot = 0; slot < td::kGroupHeads; ++slot) {
        int best = -1;
        float best_abs = -1.0f;
        for (int head = 0; head < td::kHeads; ++head) {
          if (used[head]) continue;
          const float magnitude = fabsf(w[head]);
          if (magnitude > best_abs ||
              (magnitude == best_abs && (best < 0 || head < best))) {
            best = head;
            best_abs = magnitude;
          }
        }
        order[slot] = best;
        used[best] = 1;
      }
    }
    int cursor = td::kGroupHeads;
    for (int head = 0; head < td::kHeads; ++head) {
      if (!used[head]) order[cursor++] = head;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < td::kHeads * td::kHeadDim;
       idx += blockDim.x) {
    const int group = idx / (td::kHeadDim * td::kGroupHeads);
    const int within = idx % (td::kHeadDim * td::kGroupHeads);
    const int k = within / td::kGroupHeads;
    const int group_head = within % td::kGroupHeads;
    const int head = order[group * td::kGroupHeads + group_head];
    q_packed[idx] = q[head * td::kHeadDim + k];
  }
  if (threadIdx.x < td::kHeads) {
    const int head = order[threadIdx.x];
    w_packed[threadIdx.x] = w[head];
    packed_ids[threadIdx.x] = head;
  }
}

__global__ __launch_bounds__(128) void full64_sync_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys, float* __restrict__ output,
    int length) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[td::kBlockTokens * td::kHeadDim];
  td::load_k_sync(keys, k_smem, blockIdx.x, length);
  __syncthreads();

  float score0 = 0.0f;
  float score1 = 0.0f;
#pragma unroll
  for (int group = 0; group < td::kGroups; ++group) {
    td::compute_group(k_smem, q_packed, w_packed, group, score0, score1);
  }
  const int lane = threadIdx.x & 31;
  if ((lane & 3) == 0) {
    const int warp = threadIdx.x >> 5;
    const int row = lane >> 2;
    const int token0 = blockIdx.x * td::kBlockTokens + warp * 16 + row;
    const int token1 = token0 + 8;
    if (token0 < length) output[token0] = score0;
    if (token1 < length) output[token1] = score1;
  }
}

template <int KStride, bool XorSwizzle, bool QShared>
__global__ __launch_bounds__(128) void full64_sync_variant_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys, float* __restrict__ output,
    int length) {
  extern __shared__ __align__(16) unsigned char raw_smem[];
  auto* k_smem = reinterpret_cast<__nv_bfloat16*>(raw_smem);
  auto* q_smem = k_smem + td::kBlockTokens * KStride;
  td::load_k_sync_layout<KStride, XorSwizzle>(keys, k_smem, blockIdx.x,
                                               length);
  if constexpr (QShared) {
    constexpr int q_vectors = td::kHeads * td::kHeadDim / 8;
    for (int vec = threadIdx.x; vec < q_vectors; vec += blockDim.x) {
      reinterpret_cast<uint4*>(q_smem)[vec] =
          reinterpret_cast<const uint4*>(q_packed)[vec];
    }
  }
  __syncthreads();
  const __nv_bfloat16* q_source = QShared ? q_smem : q_packed;
  float score0 = 0.0f;
  float score1 = 0.0f;
#pragma unroll
  for (int group = 0; group < td::kGroups; ++group) {
    td::compute_group_layout<KStride, XorSwizzle>(
        k_smem, q_source, w_packed, group, score0, score1);
  }
  const int lane = threadIdx.x & 31;
  if ((lane & 3) == 0) {
    const int warp = threadIdx.x >> 5;
    const int row = lane >> 2;
    const int token0 = blockIdx.x * td::kBlockTokens + warp * 16 + row;
    const int token1 = token0 + 8;
    if (token0 < length) output[token0] = score0;
    if (token1 < length) output[token1] = score1;
  }
}

__device__ __forceinline__ bool compute_h8_decision(
    const __nv_bfloat16* q_packed, const float* w_packed,
    const __nv_bfloat16* k_smem, bool precomputed, bool mask_value,
    bool online, bool direct_value, float threshold, float* scratch,
    float* block_max_out, int block, float& score0, float& score1) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp < td::kComputeWarps) {
    td::compute_group(k_smem, q_packed, w_packed, 0, score0, score1);
    float warp_max = (lane & 3) == 0
        ? fmaxf(score0, score1)
        : td::negative_infinity();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      warp_max = fmaxf(
          warp_max, __shfl_down_sync(0xffffffffu, warp_max, offset));
    }
    if (lane == 0) scratch[warp] = warp_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_max = td::negative_infinity();
#pragma unroll
    for (int warp_id = 0; warp_id < td::kComputeWarps; ++warp_id) {
      block_max = fmaxf(block_max, scratch[warp_id]);
    }
    block_max_out[block] = block_max;
    scratch[0] = precomputed ? static_cast<float>(mask_value)
                             : static_cast<float>(direct_value ||
                                                  (online && block_max >= threshold));
  }
  __syncthreads();
  return scratch[0] != 0.0f;
}

__device__ __forceinline__ void write_progressive_output(
    float* output, int length, int block, bool accepted, float score0,
    float score1) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp < td::kComputeWarps && (lane & 3) == 0) {
    const int row = lane >> 2;
    const int token0 = block * td::kBlockTokens + warp * 16 + row;
    const int token1 = token0 + 8;
    const float out0 = accepted ? score0 : td::negative_infinity();
    const float out1 = accepted ? score1 : td::negative_infinity();
    if (token0 < length) output[token0] = out0;
    if (token1 < length) output[token1] = out1;
  }
}

__global__ __launch_bounds__(128) void fused_mask_sync_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys,
    const bool* __restrict__ promotion_mask, float* __restrict__ output,
    float* __restrict__ block_max, int length) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[td::kBlockTokens * td::kHeadDim];
  td::load_k_sync(keys, k_smem, blockIdx.x, length);
  __syncthreads();
  float score0 = 0.0f;
  float score1 = 0.0f;
  td::compute_group(k_smem, q_packed, w_packed, 0, score0, score1);
  // K4 is explicitly DATAFLOW-ONLY/PRECOMPUTED-MASK.  It does not materialize
  // a verifier reduction that the input mask has already resolved.
  if (threadIdx.x == 0) block_max[blockIdx.x] = nanf("");
  const bool accepted = promotion_mask[blockIdx.x];
  if (accepted && (threadIdx.x >> 5) < td::kComputeWarps) {
#pragma unroll
    for (int group = 1; group < td::kGroups; ++group) {
      td::compute_group(k_smem, q_packed, w_packed, group, score0, score1);
    }
  }
  write_progressive_output(output, length, blockIdx.x, accepted, score0, score1);
}

__global__ __launch_bounds__(128) void fused_online_sync_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys,
    const bool* __restrict__ direct_mask, float threshold,
    float* __restrict__ output, bool* __restrict__ accepted_out,
    float* __restrict__ block_max, int length) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[td::kBlockTokens * td::kHeadDim];
  __shared__ float scratch[td::kBlockTokens];
  td::load_k_sync(keys, k_smem, blockIdx.x, length);
  __syncthreads();
  float score0 = 0.0f;
  float score1 = 0.0f;
  const bool accepted = compute_h8_decision(
      q_packed, w_packed, k_smem, false, false, true,
      direct_mask[blockIdx.x], threshold, scratch, block_max, blockIdx.x,
      score0, score1);
  if (threadIdx.x == 0) accepted_out[blockIdx.x] = accepted;
  if (accepted && (threadIdx.x >> 5) < td::kComputeWarps) {
#pragma unroll
    for (int group = 1; group < td::kGroups; ++group) {
      td::compute_group(k_smem, q_packed, w_packed, group, score0, score1);
    }
  }
  write_progressive_output(output, length, blockIdx.x, accepted, score0, score1);
}

__global__ __launch_bounds__(128) void h8_pass_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys, float* __restrict__ h8_scores,
    float* __restrict__ block_max, int length) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[td::kBlockTokens * td::kHeadDim];
  __shared__ float scratch[td::kBlockTokens];
  td::load_k_sync(keys, k_smem, blockIdx.x, length);
  __syncthreads();
  float score0 = 0.0f;
  float score1 = 0.0f;
  td::compute_group(k_smem, q_packed, w_packed, 0, score0, score1);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if ((lane & 3) == 0) {
    const int row = lane >> 2;
    const int local0 = warp * 16 + row;
    const int local1 = local0 + 8;
    const int token0 = blockIdx.x * td::kBlockTokens + local0;
    const int token1 = blockIdx.x * td::kBlockTokens + local1;
    scratch[local0] = token0 < length ? score0 : td::negative_infinity();
    scratch[local1] = token1 < length ? score1 : td::negative_infinity();
    if (token0 < length) h8_scores[token0] = score0;
    if (token1 < length) h8_scores[token1] = score1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = td::negative_infinity();
#pragma unroll
    for (int row = 0; row < td::kBlockTokens; ++row) {
      maximum = fmaxf(maximum, scratch[row]);
    }
    block_max[blockIdx.x] = maximum;
  }
}

__global__ __launch_bounds__(128) void h56_pass_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys,
    const bool* __restrict__ promotion_mask,
    const float* __restrict__ h8_scores, float* __restrict__ output,
    int length) {
  const int block = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = lane >> 2;
  const int token0 = block * td::kBlockTokens + warp * 16 + row;
  const int token1 = token0 + 8;
  if (!promotion_mask[block]) {
    if ((lane & 3) == 0) {
      if (token0 < length) output[token0] = td::negative_infinity();
      if (token1 < length) output[token1] = td::negative_infinity();
    }
    return;
  }
  __shared__ __align__(16) __nv_bfloat16 k_smem[td::kBlockTokens * td::kHeadDim];
  td::load_k_sync(keys, k_smem, block, length);
  __syncthreads();
  float score0 = (lane & 3) == 0 && token0 < length ? h8_scores[token0] : 0.0f;
  float score1 = (lane & 3) == 0 && token1 < length ? h8_scores[token1] : 0.0f;
#pragma unroll
  for (int group = 1; group < td::kGroups; ++group) {
    td::compute_group(k_smem, q_packed, w_packed, group, score0, score1);
  }
  if ((lane & 3) == 0) {
    if (token0 < length) output[token0] = score0;
    if (token1 < length) output[token1] = score1;
  }
}

__global__ void select_threshold_kernel(const float* block_max,
                                        const bool* direct_mask,
                                        float threshold, bool* accepted,
                                        int blocks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < blocks) accepted[idx] = direct_mask[idx] || block_max[idx] >= threshold;
}

template <int ProducerWarps>
__global__ void full64_pipeline_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys, float* __restrict__ output,
    int length, int blocks) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[2][td::kBlockTokens * td::kHeadDim];
  const int chunks = (blocks + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * chunks;
  const int end = min(begin + chunks, blocks);
  if (begin >= end) return;

  constexpr int kProducerThreads = ProducerWarps * 32;
  const bool copy_active = ProducerWarps == 0 || threadIdx.x >= 128;
  const int copy_thread = ProducerWarps == 0 ? threadIdx.x : threadIdx.x - 128;
  const int copy_threads = ProducerWarps == 0 ? 128 : kProducerThreads;
  if (copy_active) {
    td::prefetch_k_async(keys, k_smem[0], begin, length, copy_thread,
                         copy_threads);
    td::cp_async_wait_all();
  }
  __syncthreads();

  int stage = 0;
  for (int block = begin; block < end; ++block) {
    const bool has_next = block + 1 < end;
    if (has_next && copy_active) {
      td::prefetch_k_async(keys, k_smem[stage ^ 1], block + 1, length,
                           copy_thread, copy_threads);
    }
    float score0 = 0.0f;
    float score1 = 0.0f;
    if ((threadIdx.x >> 5) < td::kComputeWarps) {
#pragma unroll
      for (int group = 0; group < td::kGroups; ++group) {
        td::compute_group(k_smem[stage], q_packed, w_packed, group, score0,
                          score1);
      }
      write_progressive_output(output, length, block, true, score0, score1);
    }
    __syncthreads();
    if (has_next && copy_active) td::cp_async_wait_all();
    __syncthreads();
    stage ^= 1;
  }
}

template <int ProducerWarps>
__global__ void fused_online_pipeline_kernel(
    const __nv_bfloat16* __restrict__ q_packed,
    const float* __restrict__ w_packed,
    const __nv_bfloat16* __restrict__ keys,
    const bool* __restrict__ direct_mask, float threshold,
    float* __restrict__ output, bool* __restrict__ accepted_out,
    float* __restrict__ block_max, int length, int blocks) {
  __shared__ __align__(16) __nv_bfloat16 k_smem[2][td::kBlockTokens * td::kHeadDim];
  __shared__ float scratch[td::kBlockTokens];
  const int chunks = (blocks + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * chunks;
  const int end = min(begin + chunks, blocks);
  if (begin >= end) return;

  constexpr int kProducerThreads = ProducerWarps * 32;
  const bool copy_active = ProducerWarps == 0 || threadIdx.x >= 128;
  const int copy_thread = ProducerWarps == 0 ? threadIdx.x : threadIdx.x - 128;
  const int copy_threads = ProducerWarps == 0 ? 128 : kProducerThreads;
  if (copy_active) {
    td::prefetch_k_async(keys, k_smem[0], begin, length, copy_thread,
                         copy_threads);
    td::cp_async_wait_all();
  }
  __syncthreads();

  int stage = 0;
  for (int block = begin; block < end; ++block) {
    const bool has_next = block + 1 < end;
    if (has_next && copy_active) {
      td::prefetch_k_async(keys, k_smem[stage ^ 1], block + 1, length,
                           copy_thread, copy_threads);
    }
    float score0 = 0.0f;
    float score1 = 0.0f;
    const bool accepted = compute_h8_decision(
        q_packed, w_packed, k_smem[stage], false, false, true,
        direct_mask[block], threshold, scratch, block_max, block, score0,
        score1);
    if (threadIdx.x == 0) accepted_out[block] = accepted;
    if (accepted && (threadIdx.x >> 5) < td::kComputeWarps) {
#pragma unroll
      for (int group = 1; group < td::kGroups; ++group) {
        td::compute_group(k_smem[stage], q_packed, w_packed, group, score0,
                          score1);
      }
    }
    write_progressive_output(output, length, block, accepted, score0, score1);
    __syncthreads();
    if (has_next && copy_active) td::cp_async_wait_all();
    __syncthreads();
    stage ^= 1;
  }
}

template <typename Kernel>
void set_dynamic_smem(Kernel kernel) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
}

template <int ProducerWarps>
void launch_full_pipeline(const torch::Tensor& q_packed,
                          const torch::Tensor& w_packed,
                          const torch::Tensor& keys, torch::Tensor& output,
                          int grid, cudaStream_t stream) {
  constexpr int threads = (td::kComputeWarps + ProducerWarps) * 32;
  full64_pipeline_kernel<ProducerWarps><<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()),
      w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()),
      output.data_ptr<float>(), keys.size(0),
      (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens);
}

template <int ProducerWarps>
void launch_fused_pipeline(const torch::Tensor& q_packed,
                           const torch::Tensor& w_packed,
                           const torch::Tensor& keys,
                           const torch::Tensor& direct_mask, float threshold,
                           torch::Tensor& output, torch::Tensor& accepted,
                           torch::Tensor& block_max, int grid,
                           cudaStream_t stream) {
  constexpr int threads = (td::kComputeWarps + ProducerWarps) * 32;
  fused_online_pipeline_kernel<ProducerWarps><<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()),
      w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()),
      direct_mask.data_ptr<bool>(), threshold, output.data_ptr<float>(),
      accepted.data_ptr<bool>(), block_max.data_ptr<float>(), keys.size(0),
      (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens);
}

}  // namespace

void pack_qw(torch::Tensor q, torch::Tensor w, torch::Tensor fixed_heads,
             torch::Tensor q_packed, torch::Tensor w_packed,
             torch::Tensor packed_ids) {
  CHECK_CUDA(q); CHECK_CUDA(w); CHECK_CUDA(fixed_heads); CHECK_CUDA(q_packed);
  CHECK_CUDA(w_packed); CHECK_CUDA(packed_ids);
  CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(fixed_heads);
  CHECK_CONTIGUOUS(q_packed); CHECK_CONTIGUOUS(w_packed); CHECK_CONTIGUOUS(packed_ids);
  CHECK_BF16(q); CHECK_F32(w); CHECK_BF16(q_packed); CHECK_F32(w_packed);
  CHECK_I32(fixed_heads); CHECK_I32(packed_ids);
  TORCH_CHECK(q.sizes() == torch::IntArrayRef({td::kHeads, td::kHeadDim}),
              "q must be [64,128]");
  TORCH_CHECK(w.numel() == td::kHeads, "w must have 64 elements");
  TORCH_CHECK(fixed_heads.numel() == 0 || fixed_heads.numel() == 8,
              "fixed_heads must be empty or contain 8 ids");
  auto stream = current_stream(q);
  pack_qw_kernel<<<1, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), w.data_ptr<float>(),
      fixed_heads.numel() ? fixed_heads.data_ptr<int>() : nullptr,
      fixed_heads.numel() != 0,
      reinterpret_cast<__nv_bfloat16*>(q_packed.data_ptr()),
      w_packed.data_ptr<float>(), packed_ids.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void full64_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                 torch::Tensor keys, torch::Tensor output) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(output); CHECK_F32(output);
  CHECK_CONTIGUOUS(output); TORCH_CHECK(output.numel() == keys.size(0));
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  full64_sync_kernel<<<blocks, 128, 0, current_stream(keys)>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()),
      w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()),
      output.data_ptr<float>(), keys.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void full64_sync_variant(torch::Tensor q_packed, torch::Tensor w_packed,
                         torch::Tensor keys, torch::Tensor output,
                         int64_t layout_id, bool q_shared) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(output); CHECK_F32(output);
  CHECK_CONTIGUOUS(output); TORCH_CHECK(output.numel() == keys.size(0));
  TORCH_CHECK(layout_id >= 0 && layout_id <= 2, "layout_id must be 0, 1, or 2");
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  auto stream = current_stream(keys);
  const size_t q_bytes = q_shared ? td::kHeads * td::kHeadDim * sizeof(__nv_bfloat16) : 0;
#define LAUNCH_VARIANT(STRIDE, XOR_VALUE, Q_VALUE) \
  full64_sync_variant_kernel<STRIDE, XOR_VALUE, Q_VALUE><<<blocks, 128, \
      td::kBlockTokens * STRIDE * sizeof(__nv_bfloat16) + q_bytes, stream>>>( \
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()), \
      w_packed.data_ptr<float>(), \
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()), \
      output.data_ptr<float>(), keys.size(0))
  if (layout_id == 0 && !q_shared) { LAUNCH_VARIANT(128, false, false); }
  if (layout_id == 0 && q_shared) { LAUNCH_VARIANT(128, false, true); }
  if (layout_id == 1 && !q_shared) { LAUNCH_VARIANT(136, false, false); }
  if (layout_id == 1 && q_shared) { LAUNCH_VARIANT(136, false, true); }
  if (layout_id == 2 && !q_shared) { LAUNCH_VARIANT(128, true, false); }
  if (layout_id == 2 && q_shared) { LAUNCH_VARIANT(128, true, true); }
#undef LAUNCH_VARIANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void full64_pipeline(torch::Tensor q_packed, torch::Tensor w_packed,
                     torch::Tensor keys, torch::Tensor output,
                     int64_t ctas_per_sm, int64_t producer_warps) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(output); CHECK_F32(output);
  CHECK_CONTIGUOUS(output); TORCH_CHECK(output.numel() == keys.size(0));
  TORCH_CHECK(ctas_per_sm >= 1 && ctas_per_sm <= 3);
  TORCH_CHECK(producer_warps == 0 || producer_warps == 1 || producer_warps == 4);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::min(blocks, static_cast<int>(sm * ctas_per_sm));
  auto stream = current_stream(keys);
  if (producer_warps == 0) launch_full_pipeline<0>(q_packed, w_packed, keys, output, grid, stream);
  if (producer_warps == 1) launch_full_pipeline<1>(q_packed, w_packed, keys, output, grid, stream);
  if (producer_warps == 4) launch_full_pipeline<4>(q_packed, w_packed, keys, output, grid, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void h8_pass(torch::Tensor q_packed, torch::Tensor w_packed,
             torch::Tensor keys, torch::Tensor h8_scores,
             torch::Tensor block_max) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(h8_scores); CHECK_CUDA(block_max);
  CHECK_F32(h8_scores); CHECK_F32(block_max); CHECK_CONTIGUOUS(h8_scores); CHECK_CONTIGUOUS(block_max);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  TORCH_CHECK(h8_scores.numel() == keys.size(0) && block_max.numel() == blocks);
  h8_pass_kernel<<<blocks, 128, 0, current_stream(keys)>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()), w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()), h8_scores.data_ptr<float>(),
      block_max.data_ptr<float>(), keys.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void h56_pass(torch::Tensor q_packed, torch::Tensor w_packed,
              torch::Tensor keys, torch::Tensor promotion_mask,
              torch::Tensor h8_scores, torch::Tensor output) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(promotion_mask); CHECK_CUDA(h8_scores); CHECK_CUDA(output);
  CHECK_F32(h8_scores); CHECK_F32(output); CHECK_CONTIGUOUS(promotion_mask); CHECK_CONTIGUOUS(h8_scores); CHECK_CONTIGUOUS(output);
  TORCH_CHECK(promotion_mask.scalar_type() == torch::kBool);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  TORCH_CHECK(promotion_mask.numel() == blocks && h8_scores.numel() == keys.size(0) && output.numel() == keys.size(0));
  h56_pass_kernel<<<blocks, 128, 0, current_stream(keys)>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()), w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()), promotion_mask.data_ptr<bool>(),
      h8_scores.data_ptr<float>(), output.data_ptr<float>(), keys.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_threshold(torch::Tensor block_max, torch::Tensor direct_mask,
                      double threshold, torch::Tensor accepted) {
  CHECK_CUDA(block_max); CHECK_CUDA(direct_mask); CHECK_CUDA(accepted); CHECK_F32(block_max);
  CHECK_CONTIGUOUS(block_max); CHECK_CONTIGUOUS(direct_mask); CHECK_CONTIGUOUS(accepted);
  TORCH_CHECK(direct_mask.scalar_type() == torch::kBool && accepted.scalar_type() == torch::kBool);
  TORCH_CHECK(block_max.numel() == direct_mask.numel() && accepted.numel() == block_max.numel());
  const int blocks = block_max.numel();
  select_threshold_kernel<<<(blocks + 255) / 256, 256, 0, current_stream(block_max)>>>(
      block_max.data_ptr<float>(), direct_mask.data_ptr<bool>(), static_cast<float>(threshold),
      accepted.data_ptr<bool>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_mask_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                     torch::Tensor keys, torch::Tensor promotion_mask,
                     torch::Tensor output, torch::Tensor block_max) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(promotion_mask); CHECK_CUDA(output); CHECK_CUDA(block_max);
  CHECK_F32(output); CHECK_F32(block_max); CHECK_CONTIGUOUS(promotion_mask); CHECK_CONTIGUOUS(output); CHECK_CONTIGUOUS(block_max);
  TORCH_CHECK(promotion_mask.scalar_type() == torch::kBool);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  TORCH_CHECK(promotion_mask.numel() == blocks && output.numel() == keys.size(0) && block_max.numel() == blocks);
  fused_mask_sync_kernel<<<blocks, 128, 0, current_stream(keys)>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()), w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()), promotion_mask.data_ptr<bool>(),
      output.data_ptr<float>(), block_max.data_ptr<float>(), keys.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_online_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                       torch::Tensor keys, torch::Tensor direct_mask,
                       double threshold, torch::Tensor output,
                       torch::Tensor accepted, torch::Tensor block_max) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(direct_mask); CHECK_CUDA(output); CHECK_CUDA(accepted); CHECK_CUDA(block_max);
  CHECK_F32(output); CHECK_F32(block_max); CHECK_CONTIGUOUS(direct_mask); CHECK_CONTIGUOUS(output); CHECK_CONTIGUOUS(accepted); CHECK_CONTIGUOUS(block_max);
  TORCH_CHECK(direct_mask.scalar_type() == torch::kBool && accepted.scalar_type() == torch::kBool);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  TORCH_CHECK(direct_mask.numel() == blocks && accepted.numel() == blocks && block_max.numel() == blocks && output.numel() == keys.size(0));
  fused_online_sync_kernel<<<blocks, 128, 0, current_stream(keys)>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_packed.data_ptr()), w_packed.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr()), direct_mask.data_ptr<bool>(),
      static_cast<float>(threshold), output.data_ptr<float>(), accepted.data_ptr<bool>(),
      block_max.data_ptr<float>(), keys.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_online_pipeline(torch::Tensor q_packed, torch::Tensor w_packed,
                           torch::Tensor keys, torch::Tensor direct_mask,
                           double threshold, torch::Tensor output,
                           torch::Tensor accepted, torch::Tensor block_max,
                           int64_t ctas_per_sm, int64_t producer_warps) {
  check_qwk(q_packed, w_packed, keys); CHECK_CUDA(direct_mask); CHECK_CUDA(output); CHECK_CUDA(accepted); CHECK_CUDA(block_max);
  CHECK_F32(output); CHECK_F32(block_max); CHECK_CONTIGUOUS(direct_mask); CHECK_CONTIGUOUS(output); CHECK_CONTIGUOUS(accepted); CHECK_CONTIGUOUS(block_max);
  TORCH_CHECK(direct_mask.scalar_type() == torch::kBool && accepted.scalar_type() == torch::kBool);
  TORCH_CHECK(ctas_per_sm >= 1 && ctas_per_sm <= 3);
  TORCH_CHECK(producer_warps == 0 || producer_warps == 1 || producer_warps == 4);
  const int blocks = (keys.size(0) + td::kBlockTokens - 1) / td::kBlockTokens;
  TORCH_CHECK(direct_mask.numel() == blocks && accepted.numel() == blocks && block_max.numel() == blocks && output.numel() == keys.size(0));
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::min(blocks, static_cast<int>(sm * ctas_per_sm));
  auto stream = current_stream(keys);
  if (producer_warps == 0) launch_fused_pipeline<0>(q_packed, w_packed, keys, direct_mask, static_cast<float>(threshold), output, accepted, block_max, grid, stream);
  if (producer_warps == 1) launch_fused_pipeline<1>(q_packed, w_packed, keys, direct_mask, static_cast<float>(threshold), output, accepted, block_max, grid, stream);
  if (producer_warps == 4) launch_fused_pipeline<4>(q_packed, w_packed, keys, direct_mask, static_cast<float>(threshold), output, accepted, block_max, grid, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
