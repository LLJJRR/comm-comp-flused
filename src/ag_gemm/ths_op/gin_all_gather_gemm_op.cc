//===- gin_all_gather_gemm_op.cc --------------------------------- C++ ---===//

#include "ag_gemm/ths_op/gin_all_gather_gemm_op.h"

#include <stdexcept>
#include <algorithm>

#ifdef FLUX_ENABLE_GIN_AG

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include <nccl.h>
#include <nccl_device.h>

#include "ag_gemm/gin/gin_ag_gemm_comm.h"
#include "ag_gemm/ths_op/gemm_with_barrier.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

namespace bytedance::flux::ths_op {
namespace {

ncclComm_t
create_nccl_comm(std::shared_ptr<Group> const &pg) {
  ncclComm_t comm{};
  ncclUniqueId id{};
  if (pg->get_rank() == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  pg->broadcast_cpu(&id, sizeof(id), 0);
  NCCL_CHECK(ncclCommInitRank(&comm, pg->get_size(), id, pg->get_rank()));
  return comm;
}

size_t
scalar_bytes(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16: return 2;
    case c10::ScalarType::Float: return 4;
    case c10::ScalarType::Double: return 8;
    case c10::ScalarType::Char:
    case c10::ScalarType::Byte:
#if TORCH_SUPPOER_FP8
    case c10::ScalarType::Float8_e4m3fn:
    case c10::ScalarType::Float8_e5m2:
#endif
      return 1;
    default: FLUX_CHECK(false) << "unsupported GIN AG dtype: " << int(dtype);
  }
  return 0;
}

int
choose_chunks_per_rank(int local_m, int k, size_t elt_bytes, int requested) {
  FLUX_CHECK(local_m > 0 && k > 0);
  if (requested > 0) {
    FLUX_CHECK(local_m % requested == 0)
        << "chunks_per_rank must divide local M; local_m=" << local_m
        << ", chunks_per_rank=" << requested;
    return requested;
  }

  // NCCL's native GIN AG uses a 1 MiB rail chunk. Flux additionally needs
  // row-aligned chunks so GEMM can map M tiles to ready chunks exactly.
  constexpr double target = double(1u << 20);
  int best = 1;
  double best_score = std::numeric_limits<double>::infinity();
  int max_chunks = std::min(local_m, 64);
  for (int s = 1; s <= max_chunks; ++s) {
    if (local_m % s != 0) continue;
    double bytes = double(local_m / s) * double(k) * double(elt_bytes);
    // Log distance treats 512 KiB and 2 MiB symmetrically around 1 MiB.
    double score = std::abs(std::log2(std::max(bytes, 1.0) / target));
    if (score < best_score) {
      best_score = score;
      best = s;
    }
  }
  return best;
}

}  // namespace

class GinAllGatherGemmOp::Impl {
 public:
  std::shared_ptr<Group> pg;
  int rank;
  int world_size;
  int nnodes;
  int local_world_size;
  int local_m;
  int full_m;
  int n_dim;
  int k_dim;
  c10::ScalarType input_dtype;
  c10::ScalarType output_dtype;
  size_t elt_bytes;
  size_t rank_bytes;
  size_t ag_bytes;
  int chunks;
  int nblocks;

  ncclComm_t comm{};
  ncclDevComm_t dev_comm{};
  ncclWindow_t ag_window{};
  ncclWindow_t barrier_window{};
  void *ag_ptr = nullptr;
  void *barrier_ptr = nullptr;

  torch::Tensor ag_tensor;
  torch::Tensor barrier_tensor;
  torch::Tensor producer_signal;
  torch::Tensor resident_count;

  cudaStream_t comm_stream{};
  cudaEvent_t input_ready{};
  cudaEvent_t reset_done{};
  cudaEvent_t compute_done{};
  bool has_previous_compute = false;
  bool use_multimem = false;

  GemmWithBarirer gemm;

  Impl(
      std::shared_ptr<Group> pg_,
      int32_t nnodes_,
      int32_t full_m_,
      int32_t n_dim_,
      int32_t k_dim_,
      c10::ScalarType input_dtype_,
      c10::ScalarType output_dtype_,
      int32_t chunks_per_rank_,
      int32_t gin_contexts)
      : pg(std::move(pg_)),
        rank(pg->get_rank()),
        world_size(pg->get_size()),
        nnodes(nnodes_),
        local_world_size(nnodes_ > 0 ? world_size / nnodes_ : 0),
        local_m(full_m_ / world_size),
        full_m(full_m_),
        n_dim(n_dim_),
        k_dim(k_dim_),
        input_dtype(input_dtype_),
        output_dtype(output_dtype_),
        elt_bytes(scalar_bytes(input_dtype_)),
        rank_bytes(size_t(local_m) * size_t(k_dim_) * elt_bytes),
        ag_bytes(size_t(full_m_) * size_t(k_dim_) * elt_bytes),
        chunks(choose_chunks_per_rank(local_m, k_dim_, elt_bytes, chunks_per_rank_)),
        nblocks(std::max(1, std::min(chunks, gin_contexts))),
        gemm(rank, world_size, nnodes) {
    FLUX_CHECK(nnodes > 0);
    FLUX_CHECK(gin_contexts > 0);
    FLUX_CHECK(full_m > 0 && n_dim > 0 && k_dim > 0);
    FLUX_CHECK(world_size % nnodes == 0);
    FLUX_CHECK(full_m % world_size == 0);
    FLUX_CHECK(nnodes > 1) << "GIN Rail AG is intended for multi-node execution";
    FLUX_CHECK(get_arch() == ArchEnum::Sm90) << "GIN AG+GEMM currently targets SM90";
    FLUX_CHECK(input_dtype == c10::ScalarType::Half || input_dtype == c10::ScalarType::BFloat16)
        << "GIN AG+GEMM currently targets FP16/BF16 on SM90; quantized scale-gather is not wired yet";
    FLUX_CHECK(output_dtype == input_dtype)
        << "GIN AG+GEMM currently requires output_dtype == input_dtype";

    comm = create_nccl_comm(pg);

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    FLUX_CHECK(props.deviceApiSupport) << "NCCL Device API is not supported by this communicator";
    FLUX_CHECK(props.railedGinType != NCCL_GIN_TYPE_NONE)
        << "railed GIN backend is unavailable";

    ncclTeam_t lsa = ncclTeamLsa(comm);
    ncclTeam_t rail = ncclTeamRail(comm);
    FLUX_CHECK(lsa.nRanks == local_world_size)
        << "NCCL LSA size (" << lsa.nRanks << ") differs from expected local_world_size ("
        << local_world_size << ")";
    FLUX_CHECK(rail.nRanks == nnodes)
        << "NCCL rail size (" << rail.nRanks << ") differs from nnodes (" << nnodes << ")";
    // Our chunk-id mapping assumes the standard contiguous-per-node rank layout
    // used by NCCL's LSA/rail teams: local rank varies fastest, node rank next.
    FLUX_CHECK(lsa.rank == rank % local_world_size && lsa.stride == 1)
        << "unexpected NCCL LSA rank mapping: rank=" << rank
        << ", lsa.rank=" << lsa.rank << ", lsa.stride=" << lsa.stride;
    FLUX_CHECK(rail.rank == rank / local_world_size && rail.stride == local_world_size)
        << "unexpected NCCL rail rank mapping: rank=" << rank
        << ", rail.rank=" << rail.rank << ", rail.stride=" << rail.stride;

    use_multimem = props.multimemSupport;

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaMultimem = use_multimem;
    reqs.barrierCount = nblocks;
    // Request the same RAIL connectivity used by NCCL's native symmetric GIN
    // collectives. Do NOT set ginForceEnable here: in NCCL 2.30.x that legacy
    // flag rewrites the requested connection type to FULL.
    reqs.ginContextCount = nblocks;
    reqs.ginSignalCount = nblocks * rail.nRanks;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));
    FLUX_CHECK(dev_comm.ginContextCount > 0)
        << "NCCL created a Device communicator without usable GIN contexts";
    FLUX_CHECK(dev_comm.ginSignalCount >= reqs.ginSignalCount)
        << "NCCL Device communicator reserved only " << dev_comm.ginSignalCount
        << " GIN signals, but the fused AG requires " << reqs.ginSignalCount;

    NCCL_CHECK(ncclMemAlloc(&ag_ptr, ag_bytes));
    size_t barrier_bytes = size_t(world_size) * size_t(chunks) * sizeof(int32_t);
    NCCL_CHECK(ncclMemAlloc(&barrier_ptr, barrier_bytes));

    NCCL_CHECK(ncclCommWindowRegister(
        comm, ag_ptr, ag_bytes, &ag_window, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(
        comm, barrier_ptr, barrier_bytes, &barrier_window, NCCL_WIN_COLL_SYMMETRIC));
    // NCCL can return ncclSuccess with a null window when collective symmetric
    // registration is unsupported by the communicator.  Never let that turn
    // into a device-side null-window dereference.
    FLUX_CHECK(ag_window != nullptr && barrier_window != nullptr)
        << "GIN AG+GEMM requires NCCL collective symmetric window support";

    int current_device = -1;
    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, current_device));
    // block 0 intentionally waits until every communication CTA is resident
    // before releasing the GEMM.  More CTAs than SMs can deadlock that
    // handshake, so reject such a configuration explicitly.
    FLUX_CHECK(nblocks <= device_prop.multiProcessorCount)
        << "gin_contexts/chunks require " << nblocks
        << " simultaneously resident comm CTAs, but the GPU has only "
        << device_prop.multiProcessorCount << " SMs";

    auto dev = at::TensorOptions(input_dtype).device(at::kCUDA).device_index(at::cuda::current_device());
    ag_tensor = at::from_blob(ag_ptr, {full_m, k_dim}, [](void *) {}, dev);
    barrier_tensor = at::from_blob(
        barrier_ptr,
        {int64_t(world_size) * chunks},
        [](void *) {},
        at::TensorOptions(at::ScalarType::Int)
            .device(at::kCUDA)
            .device_index(at::cuda::current_device()));

    producer_signal = torch::zeros(
        {1}, at::TensorOptions(at::ScalarType::Int).device(at::kCUDA));
    resident_count = torch::zeros(
        {1}, at::TensorOptions(at::ScalarType::Int).device(at::kCUDA));

    CUDA_CHECK(cudaStreamCreateWithPriority(
        &comm_stream, cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&reset_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
  }

  ~Impl() {
    if (comm_stream) cudaStreamSynchronize(comm_stream);
    if (has_previous_compute) cudaEventSynchronize(compute_done);

    if (dev_comm.magic) NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
    if (barrier_window) NCCL_CHECK(ncclCommWindowDeregister(comm, barrier_window));
    if (ag_window) NCCL_CHECK(ncclCommWindowDeregister(comm, ag_window));
    if (barrier_ptr) NCCL_CHECK(ncclMemFree(barrier_ptr));
    if (ag_ptr) NCCL_CHECK(ncclMemFree(ag_ptr));
    if (comm) {
      ncclCommFinalize(comm);
      ncclCommDestroy(comm);
    }
    if (input_ready) cudaEventDestroy(input_ready);
    if (reset_done) cudaEventDestroy(reset_done);
    if (compute_done) cudaEventDestroy(compute_done);
    if (comm_stream) cudaStreamDestroy(comm_stream);
  }

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      bool fast_accum,
      bool transpose_weight) {
    CHECK_CUDA(input);
    CHECK_NDIM(input, 2);
    CHECK_2D(input, local_m, k_dim);
    CHECK_TYPE(input, input_dtype);
    CHECK_CUDA(weight);
    CHECK_NDIM(weight, 2);
    CHECK_TYPE(weight, input_dtype);
    if (transpose_weight) {
      CHECK_2D(weight, k_dim, n_dim);
    } else {
      CHECK_2D(weight, n_dim, k_dim);
    }

    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();

    // The caller may have produced `input` on the current compute stream. The
    // private communication stream must not read it until those producers are
    // complete.  Also record the tensor on that external stream so PyTorch's
    // caching allocator cannot recycle its storage while the async D2D staging
    // copy is still outstanding.
    CUDA_CHECK(cudaEventRecord(input_ready, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, input_ready, 0));
    input.record_stream(
        c10::cuda::getStreamFromExternal(comm_stream, at::cuda::current_device()));

    // Single-buffer safety: no rank may reset readiness state or overwrite the
    // AG window for the next forward until its own previous GEMM is done.  The
    // device world barrier in the comm kernel makes this condition global before
    // any remote write.  Keeping staging on comm_stream also orders it after the
    // previous comm kernel/flush, so we never overwrite a source segment still
    // being consumed by the old rail ring.
    if (has_previous_compute) {
      CUDA_CHECK(cudaStreamWaitEvent(comm_stream, compute_done, 0));
    }

    CUDA_CHECK(cudaMemsetAsync(
        barrier_ptr, 0, size_t(world_size) * size_t(chunks) * sizeof(int32_t), comm_stream));
    CUDA_CHECK(cudaMemsetAsync(producer_signal.data_ptr(), 0, sizeof(int32_t), comm_stream));
    CUDA_CHECK(cudaMemsetAsync(resident_count.data_ptr(), 0, sizeof(int32_t), comm_stream));

    // The GEMM stream must not evaluate the producer signal or chunk barriers
    // against values left by the previous forward.  Merely enqueueing the
    // resets on another stream is insufficient: the wait could observe the old
    // value (1) before the memset executes and pass spuriously.
    CUDA_CHECK(cudaEventRecord(reset_done, comm_stream));
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, reset_done, 0));

    // Stage local A into its final global-rank segment. GIN and LSA both use
    // this registered window, so there is no network recv -> GEMM copy later.
    void *local_dst = static_cast<char *>(ag_ptr) + size_t(rank) * rank_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        local_dst, input.data_ptr(), rank_bytes, cudaMemcpyDeviceToDevice, comm_stream));

    gin_ag::GinAgCommParams params;
    params.dev_comm = dev_comm;
    params.ag_window = ag_window;
    params.barrier_window = barrier_window;
    params.rank = rank;
    params.world_size = world_size;
    params.local_world_size = local_world_size;
    params.chunks_per_rank = chunks;
    params.nblocks = nblocks;
    params.rank_bytes = rank_bytes;
    params.chunk_bytes = rank_bytes / size_t(chunks);
    params.producer_signal = producer_signal.data_ptr<int32_t>();
    params.resident_count = resident_count.data_ptr<int32_t>();
    params.use_multimem = use_multimem;

    gin_ag::launch_gin_ag_gemm_comm(params, comm_stream);
    CUDA_CHECK(cudaGetLastError());

    // Existing Flux GEMM blocks on producer_signal until all comm CTAs are
    // resident, then waits individual chunk barriers inside the SM90 kernel.
    auto out = gemm.forward(
        ag_tensor,
        weight,
        bias,
        output,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        barrier_tensor,
        fast_accum,
        transpose_weight,
        c10::nullopt,
        producer_signal.data_ptr<int32_t>(),
        compute_stream,
        chunks);

    CUDA_CHECK(cudaEventRecord(compute_done, compute_stream));
    has_previous_compute = true;
    return out;
  }
};

GinAllGatherGemmOp::GinAllGatherGemmOp(
    std::shared_ptr<Group> tp_group,
    int32_t nnodes,
    int32_t full_m,
    int32_t n_dim,
    int32_t k_dim,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    int32_t chunks_per_rank,
    int32_t gin_contexts)
    : impl_(new Impl(
          std::move(tp_group),
          nnodes,
          full_m,
          n_dim,
          k_dim,
          input_dtype,
          output_dtype,
          chunks_per_rank,
          gin_contexts)) {}

GinAllGatherGemmOp::~GinAllGatherGemmOp() { delete impl_; }

torch::Tensor GinAllGatherGemmOp::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    bool fast_accum,
    bool transpose_weight) {
  return impl_->forward(input, weight, bias, output, fast_accum, transpose_weight);
}

torch::Tensor GinAllGatherGemmOp::gathered_input() const { return impl_->ag_tensor; }
int32_t GinAllGatherGemmOp::chunks_per_rank() const { return impl_->chunks; }

}  // namespace bytedance::flux::ths_op

#else

namespace bytedance::flux::ths_op {
class GinAllGatherGemmOp::Impl {};
GinAllGatherGemmOp::GinAllGatherGemmOp(
    std::shared_ptr<Group>, int32_t, int32_t, int32_t, int32_t,
    c10::ScalarType, c10::ScalarType, int32_t, int32_t) {
  throw std::runtime_error("Flux was built without FLUX_ENABLE_GIN_AG");
}
GinAllGatherGemmOp::~GinAllGatherGemmOp() = default;
torch::Tensor GinAllGatherGemmOp::forward(
    torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, bool, bool) {
  throw std::runtime_error("Flux was built without FLUX_ENABLE_GIN_AG");
}
torch::Tensor GinAllGatherGemmOp::gathered_input() const { return {}; }
int32_t GinAllGatherGemmOp::chunks_per_rank() const { return 0; }
}  // namespace bytedance::flux::ths_op

#endif
