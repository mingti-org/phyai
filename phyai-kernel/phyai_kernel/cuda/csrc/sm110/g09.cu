// KDA full-model fused-bias G09 candidate; preserve stock arithmetic and pipeline.
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
static_assert(BIAS == 1 && NO_C == 1);
using BiasNode =
    std::conditional_t<ORIENT,
                       cutlass::epilogue::fusion::Sm90ColBroadcast<0, GemmTile, Element, float, Stride<_1, _0, int64_t>, 8>,
                       cutlass::epilogue::fusion::Sm90RowBroadcast<0, GemmTile, Element, float, Stride<_0, _1, int64_t>, 8>>;
using Fusion = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, Element, float, cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90AccFetch, BiasNode>;
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
  if constexpr (HasBias) thread = typename Fusion::Arguments{{}, {bias, Element(0), {}}, {}};
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
