// G07 NCU-guided repeated-M CUTLASS tile with standard fused-bias epilogue.
// The adapter substitutes tile/layout/stage defines and records the exact source hash.
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#ifndef GATED_CHUNK
#define GATED_CHUNK 32
#endif
#ifndef TM
#define TM 128
#define TN 64
#define TK 64
#define ORIENT 0
#define PACKED 0
#define TWO 0
#define STAGES 0
#define STREAMK 0
#define BIAS 1
#endif
#ifndef BIAS
#define BIAS 1
#endif
/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

namespace cutlass::gemm::kernel {
template<class P, class M, class E, class S, class Enable = void>
class ThorGatedNativeGemmUniversal;

///////////////////////////////////////////////////////////////////////////////

template<class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileSchedulerTag_>
class ThorGatedNativeGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileSchedulerTag_,
                                   cute::enable_if_t<cute::disjunction_v<
                                       cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                                                         KernelTmaWarpSpecializedSm100>,
                                       cutlass::detail::is_kernel_tag_of<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                                                         KernelTmaWarpSpecializedBlockScaledSm100>>>> {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4, "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using LayoutSFA = typename cutlass::detail::LayoutSFAType<CollectiveMainloop>::type;
  using LayoutSFB = typename cutlass::detail::LayoutSFBType<CollectiveMainloop>::type;
  using ElementSF = typename cutlass::detail::ElementSFType<CollectiveMainloop>::type;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static constexpr bool IsComplex = CollectiveEpilogue::NumAccumulatorMtxs == 2;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;
  static_assert(size<0>(TileShape{}) == 256 && size<1>(TileShape{}) == 192);
  static_assert(AccumulatorPipelineStageCount == 1 && !IsOverlappingAccum && !IsComplex);
  static_assert(THOR_TMEM_COLUMNS == 256 || THOR_TMEM_COLUMNS == 512);

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = TileSchedulerTag_;
  using TileScheduler = typename detail::TileSchedulerSelector<TileSchedulerTag, ArchTag, CtaShape_MNK, ClusterShape,
                                                               SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads = NumThreadsPerWarp;         // 1 warp
  static constexpr uint32_t NumMMAThreads = NumThreadsPerWarp;           // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;  // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;  // 1 warp
  static constexpr uint32_t NumEpilogueThreads = CollectiveEpilogue::ThreadCount;
  static constexpr uint32_t NumEpilogueWarps = NumEpilogueThreads / NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock =
      NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads + NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_load_pipe_increment(CtaShape_MNK{});

  // Fixup performed for split-/stream-K is done across warps in different CTAs
  // at epilogue tile_slice granularity. Thus, there must be one barrier per sub-tile per
  // epilogue warp.
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  // Pipeline and pipeline state types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
                                            cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) AccumulatorPipelineStorage accumulator;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Host facing host arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel device entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{};
    ElementD* gated_output = nullptr;
    float const* gated_table = nullptr;
  };

  enum class WarpCategory : int32_t { MMA = 0, Sched = 1, MainloopLoad = 2, EpilogueLoad = 3, Epilogue = 4 };

  struct IsParticipant {
    uint32_t mma = false;
    uint32_t sched = false;
    uint32_t main_load = false;
    uint32_t epi_load = false;
    uint32_t epilogue = false;
  };

  //
  // Methods
  //

  // Convert to underlying arguments.
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    (void)workspace;
    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count != 0) {
      CUTLASS_TRACE_HOST("  WARNING: SM100 tile scheduler does not allow for user specified SM counts.\n"
                         "  To restrict a kernel's resource usage, consider using CUDA driver APIs instead (green contexts).");
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles,
        CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    return {args.mode,
            args.problem_shape,
            CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace, args.hw_info),
            CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
            TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
                                                   args.hw_info, args.scheduler, scheduler_workspace),
            args.hw_info,
            args.epilogue.ptr_D,
            reinterpret_cast<float const*>(args.epilogue.ptr_C)};
  }

  static bool can_implement(Arguments const& args) {
    bool implementable =
        (args.mode == GemmUniversalMode::kGemm) or (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape,
                                                                                         args.hw_info.cluster_shape_fallback);
    }

    constexpr bool IsBlockscaled = !cute::is_void_v<ElementSF>;
    if constexpr (IsBlockscaled) {
      if constexpr (IsDynamicCluster) {
        implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape,
                                                                                           args.hw_info.cluster_shape_fallback);
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast
        // among more than 4 CTAs
        implementable &= (args.hw_info.cluster_shape.x <= 4 && args.hw_info.cluster_shape.y <= 4
                          && args.hw_info.cluster_shape_fallback.x <= 4 && args.hw_info.cluster_shape_fallback.y <= 4);
      } else {
        // Special cluster shape check for scale factor multicasts. Due to limited size of scale factors, we can't multicast
        // among more than 4 CTAs
        implementable &= ((size<0>(ClusterShape{}) <= 4) && (size<1>(ClusterShape{}) <= 4));
      }
    }

    return implementable;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles,
        CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset,
                                                      stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) { return status; }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers,
        NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
        args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles,
        CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) { return status; }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    // NOTE cluster_shape here is the major cluster shape, not fallback one
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_grid_shape(params.scheduler, problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, cluster_shape,
                                         params.hw_info);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_MNKL;

    // Account for more than one epilogue warp
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category =
        warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx) : WarpCategory::Epilogue;

    uint32_t lane_predicate = cute::elect_one_sync();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    int cluster_size = size(cluster_shape);
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_first_cta_in_cluster = cta_rank_in_cluster == 0;
    int cta_coord_v = cta_rank_in_cluster % size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;
    constexpr bool has_mma_peer_cta = size(AtomThrShapeMNK{}) == 2;
    [[maybe_unused]] uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_category == WarpCategory::Sched) && lane_predicate) { collective_mainloop.prefetch_tma_descriptors(); }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      collective_epilogue.prefetch_tma_descriptors(params.epilogue);
    }

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant = {
        (warp_category == WarpCategory::MMA),                                 // mma
        (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
        (warp_category == WarpCategory::MainloopLoad),                        // main_load
        (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed,  // epi_load
        (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) { mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer; }
    mainloop_pipeline_params.is_leader = lane_predicate && is_mma_leader_cta && is_participant.main_load;
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, cluster_shape,
                                       cute::true_type{},    // Perform barrier init
                                       cute::false_type{});  // Delay mask calculation

    // Epilogue Load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) { epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer; }
    epi_load_pipeline_params.dst_blockid = cta_rank_in_cluster;
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.initializing_warp = 1;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    typename LoadOrderBarrier::Params load_order_barrier_params;
    load_order_barrier_params.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    load_order_barrier_params.group_size = NumMainloopLoadThreads;
    load_order_barrier_params.initializing_warp = 3;
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    } else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.consumer_arv_count =
        NumSchedThreads + cluster_size * (NumMainloopLoadThreads + NumEpilogueThreads + NumMMAThreads);
    if (is_epi_load_needed) { clc_pipeline_params.consumer_arv_count += cluster_size * NumEpilogueLoadThreads; }
    clc_pipeline_params.transaction_bytes = CLCResponseSize;
    clc_pipeline_params.initializing_warp = 4;
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_pipeline_params, cluster_shape);

    // Mainloop-Epilogue pipeline
    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count = size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    accumulator_pipeline_params.initializing_warp = 5;
    AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator, accumulator_pipeline_params, cluster_shape,
                                             cute::true_type{},    // Perform barrier init
                                             cute::false_type{});  // Delay mask calculation

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    clc_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
    clc_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
    clc_throttle_pipeline_params.dst_blockid = 0;
    clc_throttle_pipeline_params.initializing_warp = 3;
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_pipeline_params);
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between MMA and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumMMAThreads + NumEpilogueThreads,
                                                      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category) {
      if constexpr (!IsOverlappingAccum) {
        if (has_mma_peer_cta && lane_predicate) { tmem_deallocation_result_barrier.init(NumMMAThreads); }
      } else {
        if (has_mma_peer_cta && lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads * 2);
        } else if (lane_predicate) {
          tmem_deallocation_result_barrier.init(NumEpilogueThreads);
        }
      }
    }

    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, shared_storage.tensors.mainloop);

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    CLCPipelineState clc_pipe_consumer_state;
    CLCPipelineState clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    //
    // TMEM "Allocation"
    //
    auto tmem_storage = collective_mainloop.template init_tmem_tensors<EpilogueTile, IsOverlappingAccum>(EpilogueTile{});

    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_arrive = is_epi_load_needed;
      bool requires_clc_query = true;

      do {
        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter =
            scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, load_inputs.k_tiles);
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }

        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
        auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
            mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next;

        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }

        auto [mainloop_producer_state_next_, unused_] =
            collective_mainloop.load(mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
                                     k_tile_iter_next, k_tile_count - k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next_;

        // Sync warp to prevent non-participating threads entering next wave early
        __syncwarp();
        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        requires_clc_query = increment_pipe;
        if (increment_pipe) { ++clc_pipe_consumer_state; }
      } while (work_tile_info.is_valid());
      collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

    }

    else if (is_participant.sched) {
      if constexpr (IsSchedDynamicPersistent) {
        // Whether a new CLC query must be performed.
        // See comment below where this variable is updated for a description of
        // why this variable is needed.
        bool requires_clc_query = true;

        cutlass::arch::wait_on_dependent_grids();

        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;

            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          // Fetch next work tile
          auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

          // Only perform a new CLC query if we consumed a new CLC query result in
          // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
          // not consume a new CLC query response is when processing stream-K units.
          // The current stream-K scheduler uses single WorkTileInfo to track multiple
          // (potentially-partial) tiles to be computed via stream-K. In this case,
          // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
          // rather than consuming a CLC query response.
          requires_clc_query = increment_pipe;
          if (increment_pipe) { ++clc_pipe_consumer_state; }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
        clc_pipeline.producer_tail(clc_pipe_producer_state);
      }
    }

    else if (is_participant.mma) {
      // Tmem allocation sequence
      tmem_allocator.allocate(THOR_TMEM_COLUMNS, &shared_storage.tmem_base_ptr);
      __syncwarp();
      tmem_allocation_result_barrier.arrive();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      auto mma_inputs = collective_mainloop.mma_init(tmem_storage, shared_storage.tensors.mainloop);

      do {
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        if (increment_pipe) { ++clc_pipe_consumer_state; }

        // Accumulator stage slice
        int acc_stage = [&]() {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_producer_state.phase() ^ 1;
          } else {
            return accumulator_pipe_producer_state.index();
          }
        }();

        if (is_mma_leader_cta) {
          mainloop_pipe_consumer_state = collective_mainloop.mma(
              cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
              cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
              collective_mainloop.slice_accumulator(tmem_storage, acc_stage), mma_inputs, cta_coord_mnkl, k_tile_count);
          accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
        }
        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Hint on an early release of global memory resources.
      // The timing of calling this function only influences performance,
      // not functional correctness.
      cutlass::arch::launch_dependent_grids();

      // Release the right to allocate before deallocations so that the next CTA can rasterize
      tmem_allocator.release_allocation_lock();

      if constexpr (!IsOverlappingAccum) {
        // Leader MMA waits for leader + peer epilogues to release accumulator stage
        if (is_mma_leader_cta) { accumulator_pipeline.producer_tail(accumulator_pipe_producer_state); }
        // Signal to peer MMA that entire tmem allocation can be deallocated
        if constexpr (has_mma_peer_cta) {
          // Leader does wait + arrive, follower does arrive + wait
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
          tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
          tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
        }
      } else {
        tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      }

      // Free entire tmem allocation
      tmem_allocator.free(tmem_base_ptr, THOR_TMEM_COLUMNS);
    }

    else if (is_participant.epi_load) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();

      bool do_load_order_wait = true;
      bool do_tail_load = false;
      int current_wave = 0;

      do {
        bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);

        // Get current work tile and fetch next work tile
        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        work_tile_info = next_work_tile_info;

        if (increment_pipe) { ++clc_pipe_consumer_state; }

        if (compute_epilogue) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          bool reverse_epi_n = IsOverlappingAccum && (current_wave % 2 == 0);
          epi_load_pipe_producer_state = collective_epilogue.template load<IsOverlappingAccum>(
              epi_load_pipeline, epi_load_pipe_producer_state, problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl, TileShape{},
              TiledMma{}, shared_storage.tensors.epilogue, reverse_epi_n);

          do_tail_load = true;
        }
        current_wave++;

        // Calculate the cta coordinates of the next work tile
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      } while (work_tile_info.is_valid());

      // Only perform a tail load if one of the work units processed performed
      // an epilogue load. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_load) {
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state, epi_store_pipeline,
                                      epi_store_pipe_producer_state);
      }
    }

    else if (is_participant.epilogue) {
      // Wait for tmem allocate here
      tmem_allocation_result_barrier.arrive_and_wait();
      uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
      collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

      bool do_tail_store = false;
      do {
        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        if (increment_pipe) { ++clc_pipe_consumer_state; }

        // Accumulator stage slice
        int acc_stage = [&]() {
          if constexpr (IsOverlappingAccum) {
            return accumulator_pipe_consumer_state.phase();
          } else {
            return accumulator_pipe_consumer_state.index();
          }
        }();

        auto accumulator = get<0>(collective_mainloop.slice_accumulator(tmem_storage, acc_stage));
        accumulator_pipe_consumer_state =
            scheduler.template fixup<IsComplex>(TiledMma{}, work_tile_info, accumulator, accumulator_pipeline,
                                                accumulator_pipe_consumer_state, typename CollectiveEpilogue::CopyOpT2R{});

        //
        // Epilogue and write to gD
        //
        if (scheduler.compute_epilogue(work_tile_info)) {
          auto [load_state_next, store_state_next, acc_state_next] =
              collective_epilogue.template store_gated<IsOverlappingAccum>(
                  params.gated_output, params.gated_table, epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                  epi_store_pipe_producer_state, accumulator_pipeline, accumulator_pipe_consumer_state, problem_shape_MNKL,
                  CtaShape_MNK{}, cta_coord_mnkl, TileShape{}, TiledMma{}, accumulator, shared_storage.tensors.epilogue);
          epi_load_pipe_consumer_state = load_state_next;
          epi_store_pipe_producer_state = store_state_next;
          accumulator_pipe_consumer_state = acc_state_next;
          do_tail_store = false;
        }
        work_tile_info = next_work_tile_info;
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

      } while (work_tile_info.is_valid());

      if constexpr (IsOverlappingAccum) {
        // Signal to peer MMA that Full TMEM alloc can be deallocated
        if constexpr (has_mma_peer_cta) { tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank); }
        tmem_deallocation_result_barrier.arrive();
      }

      // Only perform a tail store if one of the work units processed performed
      // an epilogue. An example of a case in which a tail load should not be
      // performed is in split-K if a cluster is only assigned non-final splits (for which
      // the cluster does not compute the epilogue).
      if (do_tail_store) {
        collective_epilogue.store_tail(epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline,
                                       epi_store_pipe_producer_state, CtaShape_MNK{});
      }
    }

    else {
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel
using namespace cute;
using Element = cutlass::bfloat16_t;
#ifndef NO_C
#define NO_C 0
#endif
using ElementSource = std::conditional_t<NO_C, void, Element>;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using LA = std::conditional_t<ORIENT && PACKED, Col, Row>;
using LB = std::conditional_t<!ORIENT && PACKED, Row, Col>;
using LD = std::conditional_t<ORIENT, Col, Row>;
using GemmTile = Shape<Int<TM>, Int<TN>, Int<TK>>;
#ifndef CM
#define CM (TWO ? 2 : 1)
#endif
#ifndef CN
#define CN 1
#endif
using Cluster = Shape<Int<CM>, Int<CN>, _1>;
// All supported model paths have alpha=1,beta=0; convert accumulated FP32 directly.
// This removes runtime scalar broadcasts and FFMA with unit/zero coefficients.
static_assert(BIAS == 0 && NO_C == 1);
using Fusion = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, Element, float,
                                           cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
using ES = std::conditional_t<TWO, cutlass::epilogue::TmaWarpSpecialized2Sm, cutlass::epilogue::TmaWarpSpecialized1Sm>;
using MS =
    std::conditional_t<TWO, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
using BaseEpi =
    typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, GemmTile,
                                                              Cluster, Shape<_128, _32>, float, float, ElementSource, LD, 8,
                                                              Element, LD, 8, ES, Fusion>::CollectiveOp;
// Reassemble repeated MMA-M fragments for the standard epilogue tile_slice loop.
struct Epi : BaseEpi {
  using BaseEpi::BaseEpi;
  template<bool ReuseTmem = false, class AP, class APS, class PS, class CT, class CC, class MT, class MMA, class AE, class AL>
  CUTLASS_DEVICE auto store_gated(Element* output, float const* table, LoadPipeline lp, LoadPipelineState ls, StorePipeline sp,
                                  StorePipelineState ss, AP ap, APS as, PS ps, CT ct, CC cc, MT mt, MMA mma,
                                  cute::Tensor<AE, AL> acc, TensorStorage& shared) {
    static_assert(!ReuseTmem && TM == 256 && ORIENT == 1 && TN % GATED_CHUNK == 0);
    auto acc_shape = acc.shape();
    auto acc_stride = acc.stride();
    auto matrix = make_tensor(acc.data(), make_layout(make_shape(get<0, 0>(acc_shape), get<0, 1>(acc_shape)),
                                                      make_stride(get<0, 0>(acc_stride), get<0, 1>(acc_stride))));
    using GatedTile = Shape<_128, Int<GATED_CHUNK>>;
    auto acc_tiles = flat_divide(matrix, GatedTile{});
    auto coordinate_tiles = flat_divide(make_identity_tensor(shape(matrix)), GatedTile{});
    auto tiled_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b16x{}, acc_tiles(_, _, _0{}, _0{}));
    auto thread_copy = tiled_copy.get_slice(threadIdx.x % BaseEpi::ThreadCount);
    auto source = thread_copy.partition_S(acc_tiles);
    auto coordinates = thread_copy.partition_D(coordinate_tiles);
    auto values = make_tensor<float>(shape(coordinates(_, _, _, _0{}, _0{})));
    auto wait_token = ap.consumer_try_wait(as);
    ap.consumer_wait(as, wait_token);
    CUTLASS_PRAGMA_UNROLL
    for (int tile_slice = 0; tile_slice < TN / GATED_CHUNK; ++tile_slice) {
      copy(tiled_copy, source(_, _, _, _0{}, tile_slice), values);
      cutlass::arch::fence_view_async_tmem_load();
      if (tile_slice == TN / GATED_CHUNK - 1) {
        ap.consumer_release(as);
        ++as;
      }
      auto locations = coordinates(_, _, _, _0{}, tile_slice);
      CUTLASS_PRAGMA_UNROLL
      for (int element = 0; element < size(values); ++element) {
        float gate_accumulator = values(element);
        float up_accumulator = __shfl_xor_sync(0xffffffff, gate_accumulator, 1);
        auto location = locations(element);
        int packed_row = get<0>(cc) * size<0>(ct) + get<0>(location);
        int token = get<1>(cc) * size<1>(ct) + get<1>(location);
        if (packed_row < get<0>(ps) && token < get<1>(ps) && packed_row % 2 == 0) {
          auto gate = cutlass::NumericConverter<Element, float>{}(gate_accumulator);
          auto up = cutlass::NumericConverter<Element, float>{}(up_accumulator);
          float gelu = table[gate.raw()];
          float expanded_up = float(up);
          float activated;
          asm("mul.ftz.f32 %0,%1,%2;" : "=f"(activated) : "f"(gelu), "f"(expanded_up));
          output[int64_t(token) * (get<0>(ps) / 2) + packed_row / 2] = cutlass::NumericConverter<Element, float>{}(activated);
        }
      }
    }
    return cute::make_tuple(ls, ss, as);
  }
  template<bool ReuseTmem = false, class AP, class APS, class PS, class CT, class CC, class MT, class MMA, class AE, class AL>
  CUTLASS_DEVICE auto store(LoadPipeline lp, LoadPipelineState ls, StorePipeline sp, StorePipelineState ss, AP ap, APS as,
                            PS ps, CT ct, CC cc, MT mt, MMA mma, cute::Tensor<AE, AL> acc, TensorStorage& shared) {
    static_assert(!ReuseTmem);
    auto s = acc.shape();
    auto d = acc.stride();
    auto mn_shape = make_shape(make_shape(get<0, 0>(s), get<1>(s)), get<0, 1>(s));
    auto mn_stride = make_stride(make_stride(get<0, 0>(d), get<1>(d)), get<0, 1>(d));
    auto folded = make_tensor(acc.data(), make_layout(make_shape(mn_shape, _1{}, _1{}), make_stride(mn_stride, _0{}, _0{})));
    return BaseEpi::template store<false>(lp, ls, sp, ss, ap, as, ps, ct, cc, mt, mma, folded, shared);
  }
};
using Stage =
    std::conditional_t<STAGES == 0, cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
                       cutlass::gemm::collective::StageCount<STAGES>>;
// Repeat native 128-row-per-SM atoms. Larger tiles mirror measured FlashRT geometry.
static_assert((TWO && TM % 256 == 0) || (!TWO && TM % 128 == 0));
constexpr int RepeatM = TM / (TWO ? 256 : 128);
using AtomTile = Shape<Int<TWO ? 256 : 128>, Int<TN>, _64>;
using AtomMain =
    typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, LA, 8,
                                                          Element, LB, 8, float, AtomTile, Cluster,
                                                          cutlass::gemm::collective::StageCount<STAGES>, MS>::CollectiveOp;
// Domain = (native128, CTA, repetition); each CTA owns contiguous128*RepeatM rows.
using RepeatedMma = decltype(make_tiled_mma(
    typename AtomMain::TiledMma::Atom{}, Layout<Shape<_1, _1, _1>>{},
    make_tile(Layout<Shape<_128, Int<TWO ? 2 : 1>, Int<RepeatM>>, Stride<_1, Int<128 * RepeatM>, _128>>{}, _, _)));
using RepeatedPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecialized<STAGES, 2, ACC_STAGES, Cluster, cutlass::arch::Sm100>;
using Main = cutlass::gemm::collective::CollectiveMma<
    RepeatedPolicy, GemmTile, Element, typename AtomMain::StrideA, Element, typename AtomMain::StrideB, RepeatedMma,
    typename AtomMain::GmemTiledCopyA, typename AtomMain::SmemLayoutAtomA, typename AtomMain::SmemCopyAtomA,
    typename AtomMain::TransformA, typename AtomMain::GmemTiledCopyB, typename AtomMain::SmemLayoutAtomB,
    typename AtomMain::SmemCopyAtomB, typename AtomMain::TransformB>;
// Dense arbitrary-width N groups avoid SM100 residual-strip fallback.
struct ThorDenseSchedulerTag {};
namespace cutlass::gemm::kernel::detail {
template<class CS, uint32_t S>
struct ThorDenseScheduler : PersistentTileSchedulerSm100<CS, S> {
  using Base = PersistentTileSchedulerSm100<CS, S>;
  using typename Base::CLCResponse;
  using typename Base::Params;
  using typename Base::WorkTileInfo;
  Params const& p_;
  template<class... Args>
  static Params to_underlying_arguments(Args&&... args) {
    Params p = Base::to_underlying_arguments(std::forward<Args>(args)...);
    p.log_swizzle_size_ = p.divmod_swizzle_size_.divisor > 0 ? p.divmod_swizzle_size_.divisor : 1;
    p.divmod_swizzle_size_.divisor = 0;
    return p;
  }
  CUTLASS_DEVICE ThorDenseScheduler(CLCResponse* r, Params const& p, dim3 bid) : Base(r, p, bid), p_(p) {}
  CUTLASS_DEVICE auto work_tile_to_cta_coord(WorkTileInfo w) {
    int cm = size<0>(CS{}), cn = size<1>(CS{});
    int nm = p_.problem_tiles_m_, nn = p_.problem_tiles_n_;
    int wm = w.M_idx, wn = w.N_idx;
    if (w.is_valid()) {
      int linear = (wm / cm) + nm * (wn / cn);
      int width = p_.log_swizzle_size_;
      int group = linear / (nm * width);
      int first = group * width;
      int actual = cute::min(width, nn - first);
      int offset = linear - group * nm * width;
      int mi = offset / actual, ni = first + offset % actual;
      if (group % 2) mi = nm - 1 - mi;
      wm = mi * cm + wm % cm;
      wn = ni * cn + wn % cn;
    }
    return make_coord(wm, wn, _, w.L_idx);
  }
};
template<class TS, class CS, uint32_t S>
struct TileSchedulerSelector<ThorDenseSchedulerTag, cutlass::arch::Sm100, TS, CS, S> {
  using Scheduler = ThorDenseScheduler<CS, S>;
};
}  // namespace cutlass::gemm::kernel::detail
using Scheduler = ThorDenseSchedulerTag;
using Kernel = cutlass::gemm::kernel::ThorGatedNativeGemmUniversal<Shape<int, int, int, int>, Main, Epi, Scheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
thread_local std::string last_error;
static void ck(cutlass::Status s) {
  if (s != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(s));
}
static void cu(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
template<bool HasBias, class ThreadArgs>
static void set_bias(ThreadArgs& thread, Element* bias) {
  if constexpr (HasBias) thread.bias_ptr = bias;
}
struct Plan {
  int m, n, k;
  void* workspace = nullptr;
  size_t bytes = 0;
  Gemm gemm;
  Gemm::Arguments args;
  std::string info;
  Plan(int m_, int n_, int k_, void* x, void* w, void* b, void* y, cudaStream_t stream, int swizzle = 8, int raster = 0)
      : m(m_), n(n_), k(k_) {
    int mm = ORIENT ? n : m, nn = ORIENT ? m : n;
    auto sa = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(mm, k, 1));
    auto sb = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(nn, k, 1));
    auto sd = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, make_shape(mm, nn, 1));
    int active_device = 0;
    cu(cudaGetDevice(&active_device));
    int sms = 0;
    cu(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, active_device));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = active_device;
    hw.sm_count = sms;
    args = {cutlass::gemm::GemmUniversalMode::kGemm,
            {mm, nn, k, 1},
            {(Element*)(ORIENT ? w : x), sa, (Element*)(ORIENT ? x : w), sb},
            {{}, b, sd, (Element*)y, sd},
            hw};
    if ((m != 528 && m != 560 && m != 624 && m != 712) || n != 32768 || k != 2048 || !b)
      throw std::runtime_error("Invalid gated G14 contract");
    // NCU-guided isolated test: group neighboring output tiles for L2 reuse.
    if (swizzle < 0 || swizzle > 32 || raster < 0 || raster > 2) throw std::runtime_error("Invalid scheduler configuration");
    args.scheduler.max_swizzle_size = swizzle;
    using Raster = decltype(args.scheduler.raster_order);
    if (raster != 0) throw std::runtime_error("Dense scheduler requires raster0");
    args.scheduler.raster_order = Raster::AlongM;

    set_bias<BIAS>(args.epilogue.thread, (Element*)b);
    ck(Gemm::can_implement(args));
    bytes = Gemm::get_workspace_size(args);
    if (bytes) cu(cudaMalloc(&workspace, bytes));
    try {
      ck(gemm.initialize(args, workspace, stream));
    } catch (...) {
      if (workspace) cudaFree(workspace);
      throw;
    }
    auto grid = Gemm::get_grid_shape(args, workspace);
    std::ostringstream os;
    os << "{\"workspace_bytes\":" << bytes << ",\"shared_bytes\":" << Kernel::SharedStorageSize << ",\"grid\":[" << grid.x
       << "," << grid.y << "," << grid.z << "],\"tile\":[" << TM << "," << TN << "," << TK << "],\"sm_count\":" << sms
       << ",\"orientation\":" << ORIENT << ",\"packed\":" << PACKED << ",\"two_sm\":" << TWO << ",\"stages\":" << STAGES
       << ",\"streamk\":" << STREAMK << ",\"swizzle\":" << swizzle << ",\"raster\":" << raster << ",\"cluster\":[" << CM << ","
       << CN << ",1]"
       << ",\"mainloop_stages\":" << Main::DispatchPolicy::Stages
       << ",\"epilogue_shared_bytes\":" << sizeof(typename Epi::SharedStorage) << ",\"gated_chunk\":" << GATED_CHUNK
       << ",\"output_columns\":" << n / 2 << ",\"paired_rows\":true}";
    info = os.str();
  }
  ~Plan() {
    if (workspace) cudaFree(workspace);
  }
  void run(void* x, void* w, void* b, void* y, cudaStream_t stream) {
    args.mainloop.ptr_A = (Element*)(ORIENT ? w : x);
    args.mainloop.ptr_B = (Element*)(ORIENT ? x : w);
    args.epilogue.ptr_C = b;
    args.epilogue.ptr_D = (Element*)y;
    set_bias<BIAS>(args.epilogue.thread, (Element*)b);
    ck(Kernel::initialize_workspace(args, workspace, stream));
    ck(gemm.update(args, workspace));
    ck(gemm.run(stream, nullptr, true));
  }
};
extern "C" const char* ct_error() { return last_error.c_str(); }
extern "C" void* ct_create(int m, int n, int k, void* x, void* w, void* b, void* y, void* stream) {
  try {
    return new Plan(m, n, k, x, w, b, y, (cudaStream_t)stream);
  } catch (std::exception const& e) {
    last_error = e.what();
    return nullptr;
  }
}
extern "C" int ct_run(void* p, void* x, void* w, void* b, void* y, void* stream) {
  try {
    ((Plan*)p)->run(x, w, b, y, (cudaStream_t)stream);
    return 0;
  } catch (std::exception const& e) {
    last_error = e.what();
    return 1;
  }
}
extern "C" const char* ct_info(void* p) { return ((Plan*)p)->info.c_str(); }
extern "C" void ct_destroy(void* p) { delete (Plan*)p; }

extern "C" void* ct_create_ex(int m, int n, int k, void* x, void* w, void* b, void* y, void* stream, int swizzle, int raster) {
  try {
    return new Plan(m, n, k, x, w, b, y, (cudaStream_t)stream, swizzle, raster);
  } catch (std::exception const& e) {
    last_error = e.what();
    return nullptr;
  }
}

#include "gemm_ffi.cuh"
