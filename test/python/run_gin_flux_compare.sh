#!/usr/bin/env bash
# Run the fair AG/RS comparison suite.
#
# Typical 2-host / 1-H100-per-host usage (run on BOTH hosts):
#   source ./flux_env.sh
#   NNODES=2 NPROC_PER_NODE=1 NODE_RANK=0 MASTER_ADDR=<host0-ip> ./test/python/run_gin_flux_compare.sh
#   NNODES=2 NPROC_PER_NODE=1 NODE_RANK=1 MASTER_ADDR=<host0-ip> ./test/python/run_gin_flux_compare.sh
#
# A single host with 2 H100s is valid for native Flux/PyTorch baselines, but the
# current GIN Rail implementation requires real NCCL LSA/rail multi-node teams.
# Set SKIP_GIN=1 for that baseline-only case.
set -euo pipefail

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
M=${M:-4096}
N=${N:-4096}
K=${K:-4096}
DTYPE=${DTYPE:-bf16}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
GIN_CONTEXTS=${GIN_CONTEXTS:-4}
GIN_CHUNK_BYTES=${GIN_CHUNK_BYTES:-1048576}
AG_CHUNKS_PER_RANK=${AG_CHUNKS_PER_RANK:-0}
SKIP_GIN=${SKIP_GIN:-0}

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"

GIN_ARGS=()
if [[ "$SKIP_GIN" == "1" ]]; then
  GIN_ARGS+=(--skip-gin)
elif [[ "$NNODES" -le 1 ]]; then
  echo "ERROR: current GIN Rail AG/RS requires NNODES > 1." >&2
  echo "For one host / two GPUs, set SKIP_GIN=1 to measure only PyTorch and native Flux." >&2
  exit 2
fi

run_torchrun() {
  local port=$1
  shift
  torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$NPROC_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$port" \
    "$@"
}

echo "========== AG+GEMM: PyTorch vs Flux vs GIN =========="
run_torchrun "$MASTER_PORT" \
  test/python/ag_gemm/test_gin_ag_vs_flux.py \
  --M "$M" --N "$N" --K "$K" \
  --nnodes "$NNODES" --dtype "$DTYPE" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --chunks-per-rank "$AG_CHUNKS_PER_RANK" \
  --gin-contexts "$GIN_CONTEXTS" \
  "${GIN_ARGS[@]}"

echo "========== GEMM+RS: PyTorch vs Flux vs GIN =========="
run_torchrun "$((MASTER_PORT + 1))" \
  test/python/gemm_rs/test_gin_rs_vs_flux.py \
  --M "$M" --N "$N" --K "$K" \
  --nnodes "$NNODES" --dtype "$DTYPE" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --gin-contexts "$GIN_CONTEXTS" \
  --gin-chunk-bytes "$GIN_CHUNK_BYTES" \
  "${GIN_ARGS[@]}"
