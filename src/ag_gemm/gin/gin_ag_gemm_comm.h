//===- gin_ag_gemm_comm.h ----------------------------------------- C++ ---===//
//
// GIN Rail-Ring AllGather producer used by Flux AG+GEMM fusion.
// The communication algorithm intentionally mirrors NCCL's symmetric
// AllGather_RailRing_LsaSTMC structure, while publishing Flux chunk barriers
// as soon as each chunk becomes locally consumable by GEMM.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>

namespace bytedance::flux::gin_ag {

struct GinAgCommParams {
  ncclDevComm_t dev_comm{};
  ncclWindow_t ag_window{};
  ncclWindow_t barrier_window{};

  int rank = 0;
  int world_size = 0;
  int local_world_size = 0;
  int chunks_per_rank = 1;
  int nblocks = 1;

  size_t rank_bytes = 0;
  size_t chunk_bytes = 0;

  // GEMM stream waits until all communication CTAs are resident.
  int32_t *producer_signal = nullptr;
  int32_t *resident_count = nullptr;

  // Set when the NCCL communicator exposes NVLS/multimem. Otherwise the
  // communication kernel falls back to LSA copy for node-local dissemination.
  bool use_multimem = false;
};

void launch_gin_ag_gemm_comm(GinAgCommParams const &params, cudaStream_t stream);

}  // namespace bytedance::flux::gin_ag
