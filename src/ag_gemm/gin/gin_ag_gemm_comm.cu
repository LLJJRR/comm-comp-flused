//===- gin_ag_gemm_comm.cu ---------------------------------------- CUDA ---===//
//
// Copyright 2026.
// SPDX-License-Identifier: Apache-2.0
//
// This kernel borrows the communication structure of NCCL's
// AllGather_RailRing_LsaSTMC symmetric kernel:
//   - same-local-rank GPUs form a rail across nodes;
//   - warp 0 performs the GIN rail ring;
//   - remaining warps disseminate each arrived chunk across the LSA team;
//   - GIN signals provide remote visibility/order;
//   - gin.flush() provides sender-side source completion.
//
// Flux-specific addition: after LSA dissemination completes, a release-store
// publishes barrier[global_chunk] = 1 on every local GPU. The existing Flux
// SM90 AG GEMM kernel consumes that barrier and can start dependent M tiles
// before the full AllGather is complete.
//
//===----------------------------------------------------------------------===//

#include "ag_gemm/gin/gin_ag_gemm_comm.h"

#include <cuda/atomic>

namespace bytedance::flux::gin_ag {
namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kRingThreads = kWarpSize;

__device__ __forceinline__ size_t
chunk_nbytes(GinAgCommParams const &p, int split) {
  size_t off = size_t(split) * p.chunk_bytes;
  if (off >= p.rank_bytes) return 0;
  size_t remain = p.rank_bytes - off;
  return p.chunk_bytes < remain ? p.chunk_bytes : remain;
}

__device__ __forceinline__ size_t
chunk_offset(GinAgCommParams const &p, int global_rank, int split) {
  return size_t(global_rank) * p.rank_bytes + size_t(split) * p.chunk_bytes;
}

// Publish Flux's ready condition only after all threads participating in the
// local dissemination have completed their stores. Each LSA peer owns a local
// barrier array at the same logical window offset.
__device__ __forceinline__ void
publish_ready(
    GinAgCommParams const &p,
    ncclCoopWarpSpan warps,
    int global_rank,
    int split) {
  // Every worker may have issued a subset of the LSA/multimem stores. A
  // system fence only orders stores issued by the calling thread, therefore we
  // need a second group sync after *all* workers have executed their fence.
  // Without it, a fast worker could publish ready=1 while another worker's
  // peer stores are not yet system-visible.
  warps.sync();
  __threadfence_system();
  warps.sync();

  int chunk_id = global_rank * p.chunks_per_rank + split;
  ncclSymPtr<int32_t> ready{p.barrier_window, size_t(chunk_id) * sizeof(int32_t)};
  ncclTeam lsa = ncclTeamLsa(p.dev_comm);

  for (int peer = warps.thread_rank(); peer < lsa.nRanks; peer += warps.size()) {
    int32_t *dst = ready.lsaPtr(peer);
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> flag(*dst);
    flag.store(1, cuda::memory_order_release);
  }
  warps.sync();
}

__device__ __forceinline__ void
local_disseminate(
    GinAgCommParams const &p,
    ncclCoopWarpSpan warps,
    int global_rank,
    int split,
    size_t bytes) {
  if (bytes == 0) return;
  size_t off = chunk_offset(p, global_rank, split);
  ncclSymPtr<uint8_t> sym{p.ag_window, off};
  uint8_t *src = sym.localPtr();

  if (p.use_multimem) {
    ncclMultimemCopy<uint8_t>(warps, src, p.ag_window, off, bytes, p.dev_comm.lsaMultimem);
  } else {
    ncclLsaCopy<uint8_t>(warps, src, p.ag_window, off, bytes, ncclTeamLsa(p.dev_comm));
  }

  publish_ready(p, warps, global_rank, split);
}

__global__ void
gin_ag_gemm_comm_kernel(GinAgCommParams p) {
  ncclCoopCta cta;
  ncclTeam rail = ncclTeamRail(p.dev_comm);
  ncclGin gin(p.dev_comm, int(blockIdx.x % p.dev_comm.ginContextCount));

  // One independent signal sequence per comm block and rail peer, exactly like
  // NCCL's symmetric GIN AG. The public DevComm reserves this range starting at 0.
  ncclGinSignal_t rail_signals = ncclGinSignal_t(blockIdx.x * rail.nRanks);
  int next_peer = (rail.rank + 1) % rail.nRanks;
  int prev_peer = (rail.rank + rail.nRanks - 1) % rail.nRanks;
  uint64_t *shadow = gin.getSignalShadowPtr(rail_signals + prev_peer);
  uint64_t local_signal_value = *shadow;

  // The producer signal must not be released until every comm CTA is resident;
  // otherwise a persistent GEMM can occupy the GPU and starve the progress CTA
  // that it is waiting on.
  if (threadIdx.x == 0) atomicAdd(p.resident_count, 1);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (atomicAdd(p.resident_count, 0) != p.nblocks) {
      __nanosleep(64);
    }
    __threadfence_system();
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> ready(*p.producer_signal);
    ready.store(1, cuda::memory_order_release);
  }
  __syncthreads();

  // All ranks enter only after their previous GEMM has stopped consuming the
  // shared AG window. This makes single-buffer window reuse safe across forwards.
  ncclBarrierSession<ncclCoopCta> world_barrier(
      cta, ncclTeamTagWorld(), gin, uint32_t(blockIdx.x), p.use_multimem);
  world_barrier.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (threadIdx.x < kRingThreads) {
    ncclCoopWarpSpan warps(0, 1, 0);

    // Keep data-peer outermost, matching NCCL's RailRing implementation.
    // This matters because the network warp intentionally does not consume the
    // final predecessor signal; putting split outermost would make that leftover
    // signal alias the first wait of the next split.
    for (int step = 0; step < rail.nRanks - 1; ++step) {
      int data_peer = (rail.rank - step + rail.nRanks) % rail.nRanks;
      int global_rank = ncclTeamRankToWorld(p.dev_comm, rail, data_peer);

      for (int split = int(blockIdx.x); split < p.chunks_per_rank; split += int(gridDim.x)) {
        size_t bytes = chunk_nbytes(p, split);
        if (bytes == 0) continue;
        size_t off = chunk_offset(p, global_rank, split);

        if (data_peer != rail.rank) {
          gin.waitSignal(warps, rail_signals + prev_peer, local_signal_value + 1, 32);
          ++local_signal_value;
        }

        // For the first step source data is our own gathered-buffer segment.
        // For later steps the just-received segment is forwarded in-place.
        gin.put(
            rail,
            next_peer,
            p.ag_window,
            off,
            p.ag_window,
            off,
            bytes,
            ncclGin_SignalInc{rail_signals + rail.rank},
            ncclGin_None{},
            warps);
      }
    }

    // Sender-side source lifetime: all GIN reads from the AG window have retired.
    gin.flush(warps);
  } else {
    ncclCoopWarpSpan warps(1, blockDim.x / kWarpSize - 1, 1);

    // Same ordering as the network warp: data peer first, then all chunks
    // assigned to this block. The LSA side consumes one additional rail peer
    // (the predecessor's final data), exactly as NCCL's AG does.
    for (int step = 0; step < rail.nRanks; ++step) {
      int data_peer = (rail.rank - step + rail.nRanks) % rail.nRanks;
      int global_rank = ncclTeamRankToWorld(p.dev_comm, rail, data_peer);

      for (int split = int(blockIdx.x); split < p.chunks_per_rank; split += int(gridDim.x)) {
        size_t bytes = chunk_nbytes(p, split);
        if (bytes == 0) continue;

        if (data_peer != rail.rank) {
          gin.waitSignal(warps, rail_signals + prev_peer, local_signal_value + 1, 32);
          ++local_signal_value;
        }

        local_disseminate(p, warps, global_rank, split, bytes);
      }
    }
  }

  // The LSA workers consume the predecessor's final (non-forwarded) rail step
  // as well, while the network warp stops one step earlier.  Therefore commit
  // the worker-side value, matching NCCL's native RailRing implementation.
  if (threadIdx.x == kRingThreads) *shadow = local_signal_value;

  world_barrier.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

}  // namespace

void
launch_gin_ag_gemm_comm(GinAgCommParams const &params, cudaStream_t stream) {
  gin_ag_gemm_comm_kernel<<<params.nblocks, kThreads, 0, stream>>>(params);
}

}  // namespace bytedance::flux::gin_ag
