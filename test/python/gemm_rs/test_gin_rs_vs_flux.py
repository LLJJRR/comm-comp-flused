################################################################################
# Fair performance/correctness comparison for:
#   1) PyTorch GEMM + NCCL ReduceScatter
#   2) native Flux GEMM+RS (fused local reduction)
#   3) Flux NCCL-GIN GEMM+RS
#
# FP16/BF16 only.  All paths use the same row-parallel GEMM inputs:
#   input  [M, K/world]
#   weight [N, K/world]
# and produce [M/world, N].
################################################################################

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed


@dataclass
class Timing:
    local_mean_ms: float
    ranks_avg_ms: float
    ranks_max_ms: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare PyTorch NCCL, native Flux and GIN GEMM+ReduceScatter on identical work"
    )
    p.add_argument("--M", type=int, default=4096, help="full GEMM M before ReduceScatter")
    p.add_argument("--N", type=int, default=4096, help="full/local output N")
    p.add_argument("--K", type=int, default=4096, help="GLOBAL K; each rank uses K/world")
    p.add_argument("--nnodes", type=int, required=True)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--gin-contexts", type=int, default=4)
    p.add_argument("--gin-chunk-bytes", type=int, default=1 << 20)
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument(
        "--skip-gin",
        action="store_true",
        help="run only PyTorch and native Flux (useful on a 1-node machine)",
    )
    return p.parse_args()


def _sync(group):
    group.barrier()
    torch.cuda.synchronize()
    group.barrier()


def _rank_stats(local_ms: float, group) -> Tuple[float, float]:
    dev = torch.device("cuda", torch.cuda.current_device())
    value = torch.tensor([local_ms], dtype=torch.float64, device=dev)
    total = value.clone()
    maximum = value.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    return (total.item() / group.size(), maximum.item())


def bench(fn: Callable[[], torch.Tensor], group, warmup: int, iters: int) -> Tuple[torch.Tensor, Timing]:
    _sync(group)
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    group.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        out = fn()
        ends[i].record()
    ends[-1].synchronize()

    samples = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    local_mean = sum(samples) / len(samples)
    ranks_avg, ranks_max = _rank_stats(local_mean, group)
    group.barrier()
    return out, Timing(local_mean, ranks_avg, ranks_max)


def check_close(got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype):
    # Reduction order is not necessarily bitwise-identical between NCCL, native
    # Flux and the hierarchical GIN implementation.
    if dtype == torch.bfloat16:
        atol = rtol = 2e-2
    else:
        atol = rtol = 1e-2
    max_abs = (got.float() - ref.float()).abs().max().item()
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)
    return max_abs


def main():
    args = parse_args()
    group = initialize_distributed()
    rank = group.rank()
    world = group.size()

    if world % args.nnodes != 0:
        raise ValueError(f"world_size={world} must be divisible by nnodes={args.nnodes}")
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    detected_nnodes = world // local_world
    if not args.skip_gin and args.nnodes != detected_nnodes:
        raise ValueError(
            f"GIN topology mismatch: --nnodes={args.nnodes}, but torchrun implies "
            f"{detected_nnodes} node(s) (WORLD_SIZE={world}, LOCAL_WORLD_SIZE={local_world}). "
            "Do not fake --nnodes=2 on one host with two GPUs; GIN Rail needs real multi-node teams."
        )
    if args.M % world != 0:
        raise ValueError(f"M={args.M} must be divisible by world_size={world}")
    if args.K % world != 0:
        raise ValueError(f"K={args.K} must be divisible by world_size={world}")
    if not args.skip_gin and args.nnodes <= 1:
        raise ValueError("GIN Rail RS requires --nnodes > 1")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    device = torch.device("cuda", torch.cuda.current_device())
    local_k = args.K // world
    out_m = args.M // world

    torch.manual_seed(args.seed + rank)
    inp = torch.randn((args.M, local_k), device=device, dtype=dtype) / (args.K**0.5)
    weight = torch.randn((args.N, local_k), device=device, dtype=dtype)

    partial = torch.empty((args.M, args.N), device=device, dtype=dtype)
    ref_out = torch.empty((out_m, args.N), device=device, dtype=dtype)

    # Match Flux's own perf test: max_m is rounded up for reusable internal
    # buffers, while the actual runtime M remains args.M.
    max_m = ((args.M + 1023) // 1024) * 1024
    native = flux.GemmRS(
        group,
        args.nnodes,
        max_m,
        args.N,
        dtype,
        dtype,
        transpose_weight=False,
        fuse_reduction=True,
        ring_reduction=False,
        enable_gin_rs=False,
    )

    gin = None
    if not args.skip_gin:
        gin = flux.GemmRS(
            group,
            args.nnodes,
            max_m,
            args.N,
            dtype,
            dtype,
            transpose_weight=False,
            fuse_reduction=True,
            ring_reduction=False,
            enable_gin_rs=True,
            gin_contexts=args.gin_contexts,
            gin_chunk_bytes=args.gin_chunk_bytes,
        )

    rs_opt = flux.ReduceScatterOption()

    @torch.no_grad()
    def torch_path():
        torch.mm(inp, weight.t(), out=partial)
        dist.reduce_scatter_tensor(ref_out, partial, group=group)
        return ref_out

    @torch.no_grad()
    def native_flux_path():
        return native.forward(
            inp,
            weight,
            bias=None,
            input_scale=None,
            weight_scale=None,
            output_scale=None,
            fast_accum=False,
            reduce_scatter_option=rs_opt,
        )

    @torch.no_grad()
    def gin_path():
        return gin.forward(
            inp,
            weight,
            bias=None,
            input_scale=None,
            weight_scale=None,
            output_scale=None,
            fast_accum=False,
            reduce_scatter_option=rs_opt,
        )

    torch_ref = torch_path().clone()
    torch.cuda.synchronize()
    flux_out = native_flux_path().clone()
    torch.cuda.synchronize()
    native_err = check_close(flux_out, torch_ref, dtype)

    gin_err = None
    if gin is not None:
        gout = gin_path().clone()
        torch.cuda.synchronize()
        gin_err = check_close(gout, torch_ref, dtype)

    group.barrier()
    if rank == 0:
        print("\n=== GEMM+RS correctness ===", flush=True)
        print(f"world={world}, nnodes={args.nnodes}, local_world={local_world}", flush=True)
        print(
            f"shape: A=({args.M},{local_k}), W=({args.N},{local_k}), RS_out=({out_m},{args.N})",
            flush=True,
        )
        print(f"native Flux max_abs_err={native_err:.6g}", flush=True)
        if gin is not None:
            print(
                f"GIN max_abs_err={gin_err:.6g}, gin_contexts={args.gin_contexts}, gin_chunk_bytes={args.gin_chunk_bytes}",
                flush=True,
            )

    _, t_torch = bench(torch_path, group, args.warmup, args.iters)
    _, t_flux = bench(native_flux_path, group, args.warmup, args.iters)
    t_gin = None
    if gin is not None:
        _, t_gin = bench(gin_path, group, args.warmup, args.iters)

    if rank == 0:
        print("\n=== GEMM+RS latency (CUDA-event ms) ===", flush=True)
        print("path                         rank-avg      rank-max", flush=True)
        print(f"PyTorch GEMM + NCCL RS     {t_torch.ranks_avg_ms:9.3f}    {t_torch.ranks_max_ms:9.3f}", flush=True)
        print(f"Flux GEMM+RS              {t_flux.ranks_avg_ms:9.3f}    {t_flux.ranks_max_ms:9.3f}", flush=True)
        if t_gin is not None:
            print(f"Flux GIN GEMM+RS          {t_gin.ranks_avg_ms:9.3f}    {t_gin.ranks_max_ms:9.3f}", flush=True)
            print("\n=== GIN speedup, using rank-max latency ===", flush=True)
            print(f"vs PyTorch: {t_torch.ranks_max_ms / t_gin.ranks_max_ms:.3f}x", flush=True)
            print(f"vs Flux:    {t_flux.ranks_max_ms / t_gin.ranks_max_ms:.3f}x", flush=True)
            print(
                f"vs Flux improvement: {(t_flux.ranks_max_ms / t_gin.ranks_max_ms - 1.0) * 100.0:+.2f}%",
                flush=True,
            )

    group.barrier()


if __name__ == "__main__":
    main()
