# GIN Rail-Ring AllGather + GEMM fusion

This extension keeps Flux's SM90/CUTLASS GEMM path and replaces the AG producer with a device-side NCCL GIN pipeline derived from NCCL 2.30.x `AllGather_RailRing_LsaSTMC`.

## Data path

For each local-GPU index, same-index GPUs across nodes form an NCCL rail. A communication CTA uses one GIN context. Warp 0 forwards chunks around the rail ring. The remaining warps disseminate each locally available chunk to the LSA team using multimem when available and LSA copy otherwise. Only after dissemination completes is the corresponding Flux barrier entry release-stored to `1`. The existing Flux SM90 GEMM waits on those entries and may consume ready M chunks before the full AllGather completes.

```
rail GIN put/signal(N+1)  ||  local LSA dissemination(N)  ||  GEMM tiles(N-1)
```

The AG buffer is the final GEMM A buffer; there is no separate network receive buffer followed by a copy into GEMM input.

## Correctness invariants

1. GIN PUT and its ready signal stay on the same rail peer/context.
2. A Flux chunk barrier is published only after that chunk is locally visible on the target GPU.
3. Communication CTAs become resident before the persistent GEMM is released through `producer_signal`.
4. A reset-done CUDA event prevents the compute stream from observing stale barrier/producer values from a previous forward.
5. The next forward cannot overwrite the shared AG window until the previous local GEMM is done; the device world barrier turns that local condition into a global reuse point before any new GIN write.
6. `gin.flush()` protects sender-side source-buffer reuse; remote visibility is provided by the GIN signal ordering guarantee.
7. Communication chunk IDs and Flux `m_per_data_chunk` use the identical mapping `global_rank * chunks_per_rank + split`.

## Chunking

`chunks_per_rank=0` chooses a divisor of local M whose byte size is closest to NCCL's native 1 MiB GIN AG chunk. An explicit positive value must divide local M. Flux's SM90 cooperative/pingpong kernels now receive `chunks_per_rank` at runtime instead of using the old hard-coded `SPLIT=1`.

## Requirements

- H100/SM90 path.
- NCCL 2.30.x source tree and its matching built/installed Device API headers/library.
- A communicator reporting Device API support and RAIL GIN support.
- FP16/BF16 A and GEMM output in the current implementation.

The operator allocates AG and ready windows with `ncclMemAlloc` and registers them with `NCCL_WIN_COLL_SYMMETRIC`, because the communication path uses both GIN and LSA/multimem symmetric access.

## Build

With the matching NCCL source checked out at `3rdparty/nccl`:

```bash
NCCL_SOURCE_ROOT=/home/liujr/ljr/nccl-2.30.4-1 \
  ./build.sh --gin-ag --arch 90 --sm-cores 132
```

The build first builds the selected `NCCL_SOURCE_ROOT`, then compiles and links Flux against that exact tree's `build/local` headers and static library. `libflux_cuda_ths_op.so` resolves the GIN/Device-API host symbols from that same static archive (with the archive symbols hidden), so it does not accidentally bind the new APIs to a different NCCL already loaded by PyTorch. This also keeps NCCL's installed GDAKI/Proxy Device API headers self-contained.

## Python

```python
op = flux.GinAGKernel(
    tp_group,
    nnodes,
    full_m,
    n_dim,
    k_dim,
    torch.bfloat16,
    torch.bfloat16,
    chunks_per_rank=0,  # auto, around 1 MiB per rank chunk
    gin_contexts=4,
)
out = op.forward(local_input, weight)
```

A multi-iteration correctness test is provided at `test/python/ag_gemm/test_gin_ag_gemm.py`. It prepares references first and then submits fused forwards in back-to-back bursts without a per-forward host synchronize, specifically to exercise signal shadows, reset ordering, allocator stream lifetime, and AG-window reuse.
