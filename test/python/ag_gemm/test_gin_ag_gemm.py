################################################################################
# NCCL GIN Rail-Ring AllGather + Flux GEMM fusion correctness smoke test.
################################################################################

import argparse

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--nnodes", type=int, required=True)
    p.add_argument("--chunks-per-rank", type=int, default=0)
    p.add_argument("--gin-contexts", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    group = initialize_distributed()
    rank = group.rank()
    world = group.size()

    assert args.nnodes > 1
    assert world % args.nnodes == 0
    assert args.M % world == 0

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    local_m = args.M // world
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(1234 + rank)
    weight = torch.randn((args.N, args.K), device=device, dtype=dtype) / (args.K**0.5)

    op = flux.GinAGKernel(
        group,
        args.nnodes,
        args.M,
        args.N,
        args.K,
        dtype,
        dtype,
        args.chunks_per_rank,
        args.gin_contexts,
    )

    if rank == 0:
        print(f"GIN AG+GEMM chunks_per_rank={op.chunks_per_rank()}", flush=True)

    # Exercise cross-forward reuse without synchronizing the host between
    # forwards.  References are prepared first; then several fused forwards are
    # enqueued back-to-back so stale producer/barrier values, signal-shadow
    # mistakes and single-buffer lifetime races cannot be hidden by a per-call
    # torch.cuda.synchronize().
    burst = min(4, args.iters)
    completed = 0
    while completed < args.iters:
        count = min(burst, args.iters - completed)
        inputs, refs = [], []
        for _ in range(count):
            inp = torch.randn((local_m, args.K), device=device, dtype=dtype)
            ref_ag = torch.empty((args.M, args.K), device=device, dtype=dtype)
            dist.all_gather_into_tensor(ref_ag, inp, group=group)
            inputs.append(inp)  # retain storage across the asynchronous burst
            refs.append((ref_ag, torch.matmul(ref_ag, weight.t())))

        outputs = [op.forward(inp, weight) for inp in inputs]
        torch.cuda.synchronize()

        for j, (out, (_, ref)) in enumerate(zip(outputs, refs)):
            if dtype == torch.bfloat16:
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
            else:
                torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

        # The exposed gathered_input is a single reused window, so after a burst
        # it must equal the final forward's gathered input exactly.
        torch.testing.assert_close(op.gathered_input(), refs[-1][0], rtol=0, atol=0)
        completed += count
        group.barrier()
        if rank == 0:
            print(f"iterations 0..{completed - 1}: OK", flush=True)

    if rank == 0:
        print("GIN AG+GEMM correctness: PASS", flush=True)


if __name__ == "__main__":
    main()
