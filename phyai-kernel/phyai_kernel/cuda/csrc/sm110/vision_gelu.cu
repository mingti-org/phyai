// Use the upstream GDC waits and release in SM100/SM110 kernel.
// NCU-guided locality and compute-tile candidate for Thor.
// The adapter substitutes tile/layout/stage defines and records the exact source hash.
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "bf16_exact_gelu_v1.cuh"
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
#ifndef KEEP_A
#define KEEP_A 1
#endif
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
using Fusion = std::conditional_t<
    BIAS,
    std::conditional_t<ORIENT, cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<Bf16ExactGelu, Element, float>,
                       cutlass::epilogue::fusion::LinCombPerColBiasEltAct<Bf16ExactGelu, Element, float>>,
    cutlass::epilogue::fusion::LinearCombination<Element, float>>;
using ES = std::conditional_t<TWO, cutlass::epilogue::TmaWarpSpecialized2Sm, cutlass::epilogue::TmaWarpSpecialized1Sm>;
using MS =
    std::conditional_t<TWO, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, GemmTile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, ElementSource, LD, 8, Element, LD, 8, ES, Fusion>::CollectiveOp;
using Stage =
    std::conditional_t<STAGES == 0, cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
                       cutlass::gemm::collective::StageCount<STAGES>>;
using BaseMain =
    typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, LA, 8,
                                                          Element, LB, 8, float, GemmTile, Cluster, Stage, MS>::CollectiveOp;
// Preserve the parent's partitioning and pipeline; prioritize reused activations in L2.
struct Main : BaseMain {
  using BaseMain::BaseMain;
  template<class LoadParams, class TileCoordMNKL, class KTileIterator>
  CUTLASS_DEVICE auto load(MainloopPipeline pipeline, MainloopPipelineState state, LoadParams const& inputs,
                           TileCoordMNKL const& coord, KTileIterator iter, int count) {
    auto [unused, tAgA_mkl, tBgB_nkl, tAsA, tBsB, mask_a, mask_b] = inputs;
    Tensor tAgA = tAgA_mkl(_, get<0>(coord) / size(typename TiledMma::AtomThrID{}), _, get<3>(coord));
    Tensor tBgB = tBgB_nkl(_, get<1>(coord), _, get<3>(coord));
    auto token = pipeline.producer_try_acquire(state);
    CUTLASS_PRAGMA_NO_UNROLL
    while (count > 0) {
      pipeline.producer_acquire(state, token);
      auto* barrier = pipeline.producer_get_barrier(state);
      int write_stage = state.index();
      ++state;
      token = pipeline.producer_try_acquire(state);
      if (cute::elect_one_sync()) {
#if KEEP_A
        copy(observed_tma_load_a_->with(*barrier, mask_a, cute::TMA::CacheHintSm100::EVICT_LAST), tAgA(_, *iter),
             tAsA(_, write_stage));
#else
        copy(observed_tma_load_a_->with(*barrier, mask_a), tAgA(_, *iter), tAsA(_, write_stage));
#endif
        copy(observed_tma_load_b_->with(*barrier, mask_b), tBgB(_, *iter), tBsB(_, write_stage));
      }
      --count;
      ++iter;
    }
    return cute::make_tuple(state, iter);
  }
};
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
    if (raster != 0) throw std::runtime_error("Dense scheduler requires raster=0");
    args.scheduler.raster_order = Raster::AlongM;
    args.epilogue.thread.alpha = 1;
    args.epilogue.thread.beta = 0;
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

__global__ void pi05_bf16_exact_gelu_values(Element* output, const float* input, int elements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) { output[index] = Element(Bf16ExactGelu<float>{}(input[index])); }
}

extern "C" int ct_gelu_values(void* output, const void* input, int elements, void* stream) {
  if (!output || !input || elements <= 0) return int(cudaErrorInvalidValue);
  pi05_bf16_exact_gelu_values<<<(elements + 255) / 256, 256, 0, (cudaStream_t)stream>>>((Element*)output, (const float*)input,
                                                                                        elements);
  return int(cudaGetLastError());
}

#include "gemm_ffi.cuh"
