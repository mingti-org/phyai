// NCU followup: global-load warp version stalls on long scoreboard (ratio16).
// Copy coalesced tiles once, then feed MMA from bank-skewed shared memory.
// CUTLASS mma.sync primitive with warp-local K shards in one CTA.
// NCU G00: original TC template launches1 CTA with227KiB shared memory;
// direct TMA alternative is correct but whole-K load/compute serialization is slow.
// Here16x8 output tiles expose16 CTAs; each warp accumulates K/WARPS,
// then a single shared-memory reduction fuses bias before BF16 conversion.
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cutlass/cutlass.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/memory_sm80.h>
using Element = cutlass::bfloat16_t;
using Atom = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, Element, cutlass::layout::RowMajor, Element,
                                cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
using Pair = cutlass::Array<Element, 2>;
static_assert(SHAPE_K % (WARPS * 16) == 0);
constexpr int SmemStride = SHAPE_K + 8;
constexpr int SmemBytes = (16 + NTILE) * SmemStride * sizeof(Element);
constexpr int NAtoms = NTILE / 8;
static_assert(WARPS == 4 && (NTILE == 8 || NTILE == 16));
__global__ __launch_bounds__(WARPS * 32) void warp_splitk(const Element* __restrict__ x, const Element* __restrict__ w,
                                                          const Element* __restrict__ bias, Element* __restrict__ y, int m,
                                                          int n) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int group = lane / 4, t = lane % 4;
  int row = blockIdx.x * 16 + group, col = blockIdx.y * NTILE + group;
  extern __shared__ Element operands[];
// Whole input tile is copied once using vector cp.async. The8-BF16 skew
// gives each lane a distinct bank for the subsequent32-bit operand loads.
#pragma unroll
  for (int v = threadIdx.x; v < (16 + NTILE) * (SHAPE_K / 8); v += WARPS * 32) {
    int r = v / (SHAPE_K / 8), k = (v % (SHAPE_K / 8)) * 8;
    int global_row = r < 16 ? blockIdx.x * 16 + r : blockIdx.y * NTILE + r - 16;
    auto* source = (r < 16 ? x : w) + global_row * SHAPE_K + k;
    bool valid = r < 16 ? global_row < m : global_row < n;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(operands + r * SmemStride + k, source, valid);
    if ((v / (WARPS * 32)) % 8 == 7) cutlass::arch::cp_async_fence();
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  __syncwarp();  // Eachwarp copied preciselyits ownKquarter for everyoperandrow.
  Atom::FragmentC acc[NAtoms];
#pragma unroll
  for (int na = 0; na < NAtoms; ++na) acc[na].clear();
  Atom mma;
#pragma unroll
  for (int ki = 0; ki < SHAPE_K / WARPS; ki += 16) {
    int k = warp * (SHAPE_K / WARPS) + ki + t * 2;
    Pair a[4], b[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int mr = row + (i % 2) * 8, kk = k + (i / 2) * 8;
      a[i] = *reinterpret_cast<const Pair*>(operands + (group + (i % 2) * 8) * SmemStride + kk);
    }
#pragma unroll
    for (int na = 0; na < NAtoms; ++na) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
        b[i] = *reinterpret_cast<const Pair*>(operands + (16 + na * 8 + group) * SmemStride + k + i * 8);
      mma(acc[na], *reinterpret_cast<Atom::FragmentA*>(a), *reinterpret_cast<Atom::FragmentB*>(b), acc[na]);
    }
  }
  __shared__ float partial[WARPS][NAtoms][4][32];
#pragma unroll
  for (int na = 0; na < NAtoms; ++na) {
#pragma unroll
    for (int i = 0; i < 4; ++i) partial[warp][na][i][lane] = acc[na][i];
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int na = 0; na < NAtoms; ++na) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float sum = partial[0][na][i][lane];
#pragma unroll
        for (int wi = 1; wi < WARPS; ++wi) sum += partial[wi][na][i][lane];
        int mr = row + (i / 2) * 8, nc = blockIdx.y * NTILE + na * 8 + t * 2 + i % 2;
        if (mr < m && nc < n) y[mr * n + nc] = Element(sum + (BIAS ? float(bias[nc]) : 0.f));
      }
    }
  }
}
thread_local std::string error_text;
struct Plan {
  int m, n;
  std::string info;
  Plan(int mm, int nn, int kk, void* b) : m(mm), n(nn) {
    if (kk != SHAPE_K || (BIAS && !b)) throw std::runtime_error("shape/bias mismatch");
    auto e = cudaFuncSetAttribute(warp_splitk, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    std::ostringstream s;
    s << "{\"implementation\":\"CUTLASS mma.sync warp splitK cp.async multiN single kernel\",\"warps\":" << WARPS
      << ",\"shared_bytes\":" << (WARPS * NAtoms * 32 * 4 * 4 + SmemBytes) << ",\"grid\":[" << (m + 15) / 16 << ','
      << (n + NTILE - 1) / NTILE << ",1]}";
    info = s.str();
  }
};
extern "C" const char* ct_error() { return error_text.c_str(); }
extern "C" void* ct_create(int m, int n, int k, void*, void*, void* b, void*, void*) {
  try {
    return new Plan(m, n, k, b);
  } catch (std::exception const& e) {
    error_text = e.what();
    return nullptr;
  }
}
extern "C" int ct_run(void* p, void* x, void* w, void* b, void* y, void* stream) {
  auto& q = *static_cast<Plan*>(p);
  warp_splitk<<<dim3((q.m + 15) / 16, (q.n + NTILE - 1) / NTILE), WARPS * 32, SmemBytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<Element*>(x), static_cast<Element*>(w), static_cast<Element*>(b), static_cast<Element*>(y), q.m, q.n);
  auto e = cudaGetLastError();
  if (e != cudaSuccess) {
    error_text = cudaGetErrorString(e);
    return 1;
  }
  return 0;
}
extern "C" const char* ct_info(void* p) { return static_cast<Plan*>(p)->info.c_str(); }
extern "C" void ct_destroy(void* p) { delete static_cast<Plan*>(p); }

#include "gemm_ffi.cuh"
