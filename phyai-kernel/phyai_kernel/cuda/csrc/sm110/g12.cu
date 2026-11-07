// KDA cold prefix streaming priority and depth control.
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
// Producer load method adapted from CUTLASS sm100_mma_warpspecialized.hpp.
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
                                                              Cluster, Shape<Int<EPI_M>, Int<EPI_N>>, float, float,
                                                              ElementSource, LD, 8, Element, LD, 8, ES, Fusion>::CollectiveOp;
// Reassemble repeated MMA-M fragments for the standard epilogue tile_slice loop.
struct Epi : BaseEpi {
  using BaseEpi::BaseEpi;
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
using AtomTile = Shape<Int<TWO ? 256 : 128>, Int<TN>, Int<TK>>;
using AtomMain =
    typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, LA, 8,
                                                          Element, LB, 8, float, AtomTile, Cluster,
                                                          cutlass::gemm::collective::StageCount<STAGES>, MS>::CollectiveOp;
// Domain = (native128, CTA, repetition); each CTA owns contiguous128*RepeatM rows.
using RepeatedMma = decltype(make_tiled_mma(
    typename AtomMain::TiledMma::Atom{}, Layout<Shape<_1, _1, _1>>{},
    make_tile(Layout<Shape<_128, Int<TWO ? 2 : 1>, Int<RepeatM>>, Stride<_1, Int<128 * RepeatM>, _128>>{}, _, _)));
using RepeatedPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecialized<STAGES, 2, ACC_STAGES, Cluster, cutlass::arch::Sm100>;
using BaseMain = cutlass::gemm::collective::CollectiveMma<
    RepeatedPolicy, GemmTile, Element, typename AtomMain::StrideA, Element, typename AtomMain::StrideB, RepeatedMma,
    typename AtomMain::GmemTiledCopyA, typename AtomMain::SmemLayoutAtomA, typename AtomMain::SmemCopyAtomA,
    typename AtomMain::TransformA, typename AtomMain::GmemTiledCopyB, typename AtomMain::SmemLayoutAtomB,
    typename AtomMain::SmemCopyAtomB, typename AtomMain::TransformB>;
// Preserve the upstream producer algorithm, changing only TMA L2 priority.
struct Main : BaseMain {
  using BaseMain::BaseMain;
  using Hint = std::conditional_t<TWO, cute::TMA::CacheHintSm100, cute::TMA::CacheHintSm90>;
  template<class LoadParams, class TileCoordMNKL, class KTileIterator>
  CUTLASS_DEVICE auto load(MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state,
                           LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter,
                           int k_tile_count) {
    constexpr Hint HintA = HINT_A == 1 ? Hint::EVICT_FIRST : HINT_A == 2 ? Hint::EVICT_LAST : Hint::EVICT_NORMAL;
    constexpr Hint HintB = HINT_B == 1 ? Hint::EVICT_FIRST : HINT_B == 2 ? Hint::EVICT_LAST : Hint::EVICT_NORMAL;
    auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB, mcast_mask_a, mcast_mask_b] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
    Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

    auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

      int write_stage = mainloop_pipe_producer_state.index();
      ++mainloop_pipe_producer_state;
      barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

      if (cute::elect_one_sync()) {
        copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a, HintA), tAgA(_, *k_tile_iter), tAsA(_, write_stage));
        copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b, HintB), tBgB(_, *k_tile_iter), tBsB(_, write_stage));
      }

      --k_tile_count;
      ++k_tile_iter;
    }

    return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
  }
};
using Scheduler = std::conditional_t<STREAMK, cutlass::gemm::StreamKScheduler, cutlass::gemm::PersistentScheduler>;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, Scheduler>;
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
            {{}, (Element*)y, sd, (Element*)y, sd},
            hw};
    // NCU-guided isolated test: group neighboring output tiles for L2 reuse.
    if (swizzle < 0 || swizzle > 32 || raster < 0 || raster > 2) throw std::runtime_error("Invalid scheduler configuration");
    args.scheduler.max_swizzle_size = swizzle;
    using Raster = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = raster == 1 ? Raster::AlongM : raster == 2 ? Raster::AlongN : Raster::Heuristic;

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
       << ",\"epilogue_shared_bytes\":" << sizeof(typename Epi::SharedStorage) << "}";
    info = os.str();
  }
  ~Plan() {
    if (workspace) cudaFree(workspace);
  }
  void run(void* x, void* w, void* b, void* y, cudaStream_t stream) {
    args.mainloop.ptr_A = (Element*)(ORIENT ? w : x);
    args.mainloop.ptr_B = (Element*)(ORIENT ? x : w);
    args.epilogue.ptr_C = (Element*)y;
    args.epilogue.ptr_D = (Element*)y;
    set_bias<BIAS>(args.epilogue.thread, (Element*)b);
    ck(Kernel::initialize_workspace(args, workspace, stream));
    ck(gemm.update(args, workspace));
    ck(gemm.run(stream));
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
