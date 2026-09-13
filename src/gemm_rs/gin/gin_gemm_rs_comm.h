//===- gin_gemm_rs_comm.h ----------------------------------------- C++ ---===//
//
// GIN Rail-A2A transport/final-reduction stage for Flux SM90 GEMM+RS.
// Flux GEMM produces node-local reduced tiles into a symmetric blocked buffer;
// this kernel batches ready tiles, sends them directly to their final per-source
// slots on the destination rail peer, and reduces arrived slots into dense output.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>

namespace bytedance::flux::gin_rs {

enum class DataType : int32_t { FP16 = 0, BF16 = 1 };

struct GinRsCommParams {
  ncclDevComm_t dev_comm{};
  ncclWindow_t reduce_window{};

  void *output = nullptr;
  int32_t *ready = nullptr;

  int rank = 0;
  int world_size = 0;
  int local_world_size = 0;
  int nnodes = 0;
  int node_idx = 0;

  int m_rank = 0;
  int n_dim = 0;
  int tile_m = 0;
  int tile_n = 0;
  int tiles_m_per_rank = 0;
  int n_tiles = 0;
  int tiles_per_rank = 0;
  int tiles_per_chunk = 1;
  int chunks_per_rank = 0;
  int nblocks = 1;

  size_t element_bytes = 0;
  size_t tile_bytes = 0;
  size_t slot_bytes = 0;

  // Compute waits until all communication CTAs are resident and the initial
  // cross-rank barrier is complete. This prevents persistent GEMM from starving
  // the GIN progress CTAs it depends on.
  int32_t *producer_signal = nullptr;
  int32_t *resident_count = nullptr;
};

void launch_gin_gemm_rs_comm(
    GinRsCommParams const &params, DataType dtype, cudaStream_t stream);

}  // namespace bytedance::flux::gin_rs
