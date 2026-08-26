#include <torch/extension.h>
#include <cuda_runtime.h>

void pack_qw(torch::Tensor q, torch::Tensor w, torch::Tensor fixed_heads,
             torch::Tensor q_packed, torch::Tensor w_packed,
             torch::Tensor packed_ids);
void full64_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                 torch::Tensor keys, torch::Tensor output);
void full64_sync_variant(torch::Tensor q_packed, torch::Tensor w_packed,
                         torch::Tensor keys, torch::Tensor output,
                         int64_t layout_id, bool q_shared);
void full64_pipeline(torch::Tensor q_packed, torch::Tensor w_packed,
                     torch::Tensor keys, torch::Tensor output,
                     int64_t ctas_per_sm, int64_t producer_warps);
void h8_pass(torch::Tensor q_packed, torch::Tensor w_packed,
             torch::Tensor keys, torch::Tensor h8_scores,
             torch::Tensor block_max);
void h56_pass(torch::Tensor q_packed, torch::Tensor w_packed,
              torch::Tensor keys, torch::Tensor promotion_mask,
              torch::Tensor h8_scores, torch::Tensor output);
void select_threshold(torch::Tensor block_max, torch::Tensor direct_mask,
                      double threshold, torch::Tensor accepted);
void fused_mask_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                     torch::Tensor keys, torch::Tensor promotion_mask,
                     torch::Tensor output, torch::Tensor block_max);
void fused_online_sync(torch::Tensor q_packed, torch::Tensor w_packed,
                       torch::Tensor keys, torch::Tensor direct_mask,
                       double threshold, torch::Tensor output,
                       torch::Tensor accepted, torch::Tensor block_max);
void fused_online_pipeline(torch::Tensor q_packed, torch::Tensor w_packed,
                           torch::Tensor keys, torch::Tensor direct_mask,
                           double threshold, torch::Tensor output,
                           torch::Tensor accepted, torch::Tensor block_max,
                           int64_t ctas_per_sm, int64_t producer_warps);

pybind11::dict device_properties() {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  auto attr = [device](cudaDeviceAttr key) {
    int value = 0;
    cudaDeviceGetAttribute(&value, key, device);
    return value;
  };
  pybind11::dict result;
  result["name"] = prop.name;
  result["compute_capability_major"] = prop.major;
  result["compute_capability_minor"] = prop.minor;
  result["sm_count"] = prop.multiProcessorCount;
  result["shared_memory_per_block_bytes"] = prop.sharedMemPerBlock;
  result["shared_memory_per_block_optin_bytes"] =
      attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
  result["shared_memory_per_sm_bytes"] = prop.sharedMemPerMultiprocessor;
  result["registers_per_block"] = prop.regsPerBlock;
  result["registers_per_sm"] = prop.regsPerMultiprocessor;
  result["l2_cache_bytes"] = prop.l2CacheSize;
  result["memory_clock_khz"] = prop.memoryClockRate;
  result["sm_clock_khz"] = prop.clockRate;
  result["memory_bus_width_bits"] = prop.memoryBusWidth;
  result["max_threads_per_sm"] = prop.maxThreadsPerMultiProcessor;
  result["max_threads_per_block"] = prop.maxThreadsPerBlock;
  result["warp_size"] = prop.warpSize;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_qw", &pack_qw);
  m.def("full64_sync", &full64_sync);
  m.def("full64_sync_variant", &full64_sync_variant);
  m.def("full64_pipeline", &full64_pipeline);
  m.def("h8_pass", &h8_pass);
  m.def("h56_pass", &h56_pass);
  m.def("select_threshold", &select_threshold);
  m.def("fused_mask_sync", &fused_mask_sync);
  m.def("fused_online_sync", &fused_online_sync);
  m.def("fused_online_pipeline", &fused_online_pipeline);
  m.def("device_properties", &device_properties);
}
