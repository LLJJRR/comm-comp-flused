//===- gin_all_gather_gemm_op.h ---------------------------------- C++ ---===//
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <memory>
#include <torch/torch.h>

#include "flux/ths_op/flux_shm.h"

namespace bytedance::flux::ths_op {

// H100/SM90 GIN AllGather + GEMM fusion.
// Communication follows NCCL's hierarchical RailRing + LSA dissemination
// algorithm; computation reuses Flux's barrier-driven AG GEMM kernel.
class GinAllGatherGemmOp {
 public:
  GinAllGatherGemmOp(
      std::shared_ptr<Group> tp_group,
      int32_t nnodes,
      int32_t full_m,
      int32_t n_dim,
      int32_t k_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      int32_t chunks_per_rank = 0,
      int32_t gin_contexts = 4);
  ~GinAllGatherGemmOp();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias = c10::nullopt,
      c10::optional<torch::Tensor> output = c10::nullopt,
      bool fast_accum = false,
      bool transpose_weight = false);

  torch::Tensor gathered_input() const;
  int32_t chunks_per_rank() const;

 private:
  class Impl;
  Impl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
