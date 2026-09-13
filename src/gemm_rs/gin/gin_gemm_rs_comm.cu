//===- gin_gemm_rs_comm.cu --------------------------------------- CUDA ---===//
//
// Copyright 2026.
// SPDX-License-Identifier: Apache-2.0
//
// The transport topology follows NCCL's native GIN ReduceScatter choice:
// same-local-rank GPUs form a rail and exchange node-local reduced data with
// Rail A2A.  Unlike NCCL's standalone RS, Flux has already completed the LSA
// local reduction inside the GEMM epilogue, so this kernel deliberately skips
// NCCL's Stage-0 LSA reduction/outbox and sends the final Flux tile slots
// directly.  A single-buffer world barrier at the beginning/end makes slot
// reuse safe across forwards without an extra scratch inbox/credit layer.
//
//===----------------------------------------------------------------------===//

#include "gemm_rs/gin/gin_gemm_rs_comm.h"

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace bytedance::flux::gin_rs {
namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWorkerWarps = kThreads / kWarpSize - 1;
constexpr int kVecBytes = 16;

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <typename Coop>
__device__ __forceinline__ void wait_ready(
    Coop coop, int32_t *ready, int index) {
  if (coop.thread_rank() == 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> flag(ready[index]);
    while (flag.load(cuda::memory_order_acquire) != 1) {
      __nanosleep(64);
    }
  }
  coop.sync();
}

template <typename Coop>
__device__ __forceinline__ void wait_chunk_ready(
    Coop coop,
    GinRsCommParams const &p,
    int dst_node,
    int tile0,
    int ntiles) {
  for (int i = 0; i < ntiles; ++i) {
    wait_ready(coop, p.ready, dst_node * p.tiles_per_rank + tile0 + i);
  }
}

template <typename T>
__device__ __forceinline__ void reduce_tile_to_dense(
    ncclCoopWarpSpan workers,
    GinRsCommParams const &p,
    int tile_id) {
  constexpr int kVecElts = kVecBytes / sizeof(T);
  static_assert(kVecBytes % sizeof(T) == 0);

  int tile_m_idx = tile_id / p.n_tiles;
  int tile_n_idx = tile_id - tile_m_idx * p.n_tiles;
  size_t tile_off = size_t(tile_id) * p.tile_bytes;

  // Flux's SM90 inter-node reduce buffer is blocked/tile-major.  This is the
  // same property used by the existing nvshmemx_putmem_nbi_warp(gReduce,...)
  // and bsr_reduce kernel: each tile is contiguous, while final output is dense.
  int vec_count = (p.tile_m * p.tile_n) / kVecElts;
  for (int vec = workers.thread_rank(); vec < vec_count; vec += workers.size()) {
    alignas(kVecBytes) T accum_vec[kVecElts];
    float accum[kVecElts];
#pragma unroll
    for (int i = 0; i < kVecElts; ++i) accum[i] = 0.0f;

    for (int src_node = 0; src_node < p.nnodes; ++src_node) {
      size_t slot = size_t(p.node_idx * p.nnodes + src_node) * p.slot_bytes;
      ncclSymPtr<uint8_t> sym{p.reduce_window, slot + tile_off};
      T const *src = reinterpret_cast<T const *>(sym.localPtr()) + size_t(vec) * kVecElts;
      *reinterpret_cast<uint4 *>(accum_vec) = *reinterpret_cast<uint4 const *>(src);
#pragma unroll
      for (int i = 0; i < kVecElts; ++i) accum[i] += to_float(accum_vec[i]);
    }

#pragma unroll
    for (int i = 0; i < kVecElts; ++i) accum_vec[i] = from_float<T>(accum[i]);

    int elem0 = vec * kVecElts;
    int row_in_tile = elem0 / p.tile_n;
    int col_in_tile = elem0 - row_in_tile * p.tile_n;
    int out_row = tile_m_idx * p.tile_m + row_in_tile;
    int out_col = tile_n_idx * p.tile_n + col_in_tile;
    T *dst = reinterpret_cast<T *>(p.output) + size_t(out_row) * p.n_dim + out_col;
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<uint4 const *>(accum_vec);
  }
}

template <typename T>
__global__ void gin_gemm_rs_comm_kernel(GinRsCommParams p) {
  ncclCoopCta cta;
  ncclTeam rail = ncclTeamRail(p.dev_comm);
  ncclGin gin(p.dev_comm, int(blockIdx.x % p.dev_comm.ginContextCount));

  // One monotonic incoming signal per (comm block, source rail rank).  A source
  // reuses its signal for successive chunks because every destination observes
  // that source's sends in the same deterministic chunk order.
  ncclGinSignal_t signals = ncclGinSignal_t(blockIdx.x * rail.nRanks);

  if (threadIdx.x == 0) atomicAdd(p.resident_count, 1);
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    while (atomicAdd(p.resident_count, 0) != p.nblocks) __nanosleep(64);
  }
  __syncthreads();

  // The direct receive slots are single-buffered.  Pairing the begin/end world
  // barriers with stream ordering ensures no rank starts overwriting a slot
  // while another rank is still consuming it from the previous forward.
  ncclBarrierSession<ncclCoopCta> world_barrier(
      cta, ncclTeamTagWorld(), gin, uint32_t(blockIdx.x), /*multimem=*/false);
  world_barrier.sync(cta, cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> producer(*p.producer_signal);
    producer.store(1, cuda::memory_order_release);
  }
  __syncthreads();

  if (threadIdx.x < kWarpSize) {
    ncclCoopWarpSpan send_warp(0, 1, 0);

    for (int chunk = int(blockIdx.x); chunk < p.chunks_per_rank; chunk += int(gridDim.x)) {
      int tile0 = chunk * p.tiles_per_chunk;
      int ntiles = min(p.tiles_per_chunk, p.tiles_per_rank - tile0);
      size_t bytes = size_t(ntiles) * p.tile_bytes;

      // Send one contiguous blocked-tile chunk to every other node.  Each
      // destination gets a distinct [dst_node][src_node] slot, so there are no
      // remote writer conflicts and no network-side reduction atomics.
      for (int step = 1; step < rail.nRanks; ++step) {
        int dst_node = (rail.rank + step) % rail.nRanks;
        wait_chunk_ready(send_warp, p, dst_node, tile0, ntiles);

        size_t slot = size_t(dst_node * p.nnodes + p.node_idx) * p.slot_bytes;
        size_t off = slot + size_t(tile0) * p.tile_bytes;
        gin.put(
            rail,
            dst_node,
            p.reduce_window,
            off,
            p.reduce_window,
            off,
            bytes,
            ncclGin_SignalInc{signals + rail.rank},
            ncclGin_None{},
            send_warp);
      }
    }

    // Local source-buffer lifetime: after flush the next forward may overwrite
    // this rank's source slots once the ending world barrier has also completed.
    gin.flush(send_warp);
  } else {
    ncclCoopWarpSpan workers(1, kWorkerWarps, 1);

    for (int chunk = int(blockIdx.x); chunk < p.chunks_per_rank; chunk += int(gridDim.x)) {
      int tile0 = chunk * p.tiles_per_chunk;
      int ntiles = min(p.tiles_per_chunk, p.tiles_per_rank - tile0);

      // The local-node contribution never traverses GIN.
      wait_chunk_ready(workers, p, p.node_idx, tile0, ntiles);

      // Remote signal is ordered after the corresponding PUT.  Signal shadows
      // are monotonic across forwards, so no signal memset/reset is required.
      for (int step = 1; step < rail.nRanks; ++step) {
        int src_node = (rail.rank + step) % rail.nRanks;
        uint64_t *shadow = gin.getSignalShadowPtr(signals + src_node);
        uint64_t expected = *shadow + 1;
        gin.waitSignal(workers, signals + src_node, expected, 32);
        workers.sync();
        if (workers.thread_rank() == 0) *shadow = expected;
        workers.sync();
      }

      for (int i = 0; i < ntiles; ++i) {
        reduce_tile_to_dense<T>(workers, p, tile0 + i);
      }
      workers.sync();
    }
  }

  cta.sync();
  world_barrier.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}

}  // namespace

void launch_gin_gemm_rs_comm(
    GinRsCommParams const &params, DataType dtype, cudaStream_t stream) {
  switch (dtype) {
    case DataType::FP16:
      gin_gemm_rs_comm_kernel<__half><<<params.nblocks, kThreads, 0, stream>>>(params);
      break;
    case DataType::BF16:
      gin_gemm_rs_comm_kernel<__nv_bfloat16><<<params.nblocks, kThreads, 0, stream>>>(params);
      break;
  }
}

}  // namespace bytedance::flux::gin_rs
