// NCU G01: short_scoreboard 2.2 per issued instruction. Compare LDSM and direct global operands.
// Preserve arithmetic and tile geometry to isolate operand staging costs.
// A32xK tile is reused by4warps, each computing32x32 columns. No K reduction.
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cutlass/cutlass.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/arch/memory_sm75.h>
using Element = cutlass::bfloat16_t;
using Pair = cutlass::Array<Element, 2>;
using Atom = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, Element, cutlass::layout::RowMajor, Element,
                                cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
constexpr int Stride = SHAPE_K;
__device__ __forceinline__ int swizzled_k(int row, int col) { return col ^ (((row / (64 / SHAPE_K)) & (SHAPE_K / 8 - 1)) * 8); }
constexpr int MAtoms = TM / 16, NAtoms = TN / 32;
static_assert(TM % 16 == 0 && TN % 32 == 0);
static_assert(SHAPE_K == 32 || SHAPE_K == 64);
__global__ __launch_bounds__(128) void small_k(const Element* __restrict__ x, const Element* __restrict__ w,
                                               const Element* __restrict__ bias, Element* __restrict__ y, int m, int n) {
#if LOAD_MODE == 1
  __shared__ Element smem[TM + TN][Stride];
#endif
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
#if LOAD_MODE == 1
#pragma unroll
  for (int v = threadIdx.x; v < (TM + TN) * (SHAPE_K / 8); v += 128) {
    int row = v / (SHAPE_K / 8), k = v % (SHAPE_K / 8) * 8;
    int global_row = row < TM ? blockIdx.x * TM + row : blockIdx.y * TN + row - TM;
    const Element* src = (row < TM ? x : w) + global_row * SHAPE_K + k;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(&smem[row][swizzled_k(row, k)], src,
                                                                             row < TM ? global_row < m : global_row < n);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();
#endif
  Atom::FragmentC acc[MAtoms][NAtoms];
#pragma unroll
  for (int mi = 0; mi < MAtoms; ++mi) {
#pragma unroll
    for (int ni = 0; ni < NAtoms; ++ni) acc[mi][ni].clear();
  }
  Atom mma;
#pragma unroll
  for (int step = 0; step < SHAPE_K; step += 16) {
    Atom::FragmentA aa[MAtoms];
    Atom::FragmentB bb[NAtoms];
#pragma unroll
    for (int mi = 0; mi < MAtoms; ++mi) {
#if LOAD_MODE == 1
      cutlass::Array<unsigned, 4> bits;
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 4>(
          bits, &smem[mi * 16 + lane % 16][swizzled_k(mi * 16 + lane % 16, step + (lane / 16) * 8)]);
      aa[mi] = *reinterpret_cast<Atom::FragmentA*>(&bits);
#else
      Pair a[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = blockIdx.x * TM + mi * 16 + g + (i % 2) * 8;
        a[i].clear();
        if (row < m) a[i] = *reinterpret_cast<const Pair*>(x + row * SHAPE_K + step + t * 2 + (i / 2) * 8);
      }
      aa[mi] = *reinterpret_cast<Atom::FragmentA*>(a);
#endif
    }
#pragma unroll
    for (int ni = 0; ni < NAtoms; ++ni) {
#if LOAD_MODE == 1
      cutlass::Array<unsigned, 2> bits;
      cutlass::arch::ldsm<cutlass::layout::RowMajor, 2>(
          bits, &smem[TM + warp * (TN / 4) + ni * 8 + lane % 8]
                     [swizzled_k(TM + warp * (TN / 4) + ni * 8 + lane % 8, step + ((lane / 8) % 2) * 8)]);
      bb[ni] = *reinterpret_cast<Atom::FragmentB*>(&bits);
#else
      Pair b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int col = blockIdx.y * TN + warp * (TN / 4) + ni * 8 + g;
        b[i].clear();
        if (col < n) b[i] = *reinterpret_cast<const Pair*>(w + col * SHAPE_K + step + t * 2 + i * 8);
      }
      bb[ni] = *reinterpret_cast<Atom::FragmentB*>(b);
#endif
    }
#pragma unroll
    for (int mi = 0; mi < MAtoms; ++mi) {
#pragma unroll
      for (int ni = 0; ni < NAtoms; ++ni) mma(acc[mi][ni], aa[mi], bb[ni], acc[mi][ni]);
    }
  }
  __shared__ Element out_tile[TM][TN + 8];
#pragma unroll
  for (int mi = 0; mi < MAtoms; ++mi) {
#pragma unroll
    for (int ni = 0; ni < NAtoms; ++ni) {
#pragma unroll
      for (int ri = 0; ri < 2; ++ri) {
        int row = blockIdx.x * TM + mi * 16 + g + ri * 8, col = blockIdx.y * TN + warp * (TN / 4) + ni * 8 + t * 2;
        Pair out;
#pragma unroll
        for (int i = 0; i < 2; ++i)
          out[i] = Element(acc[mi][ni][ri * 2 + i] + (BIAS && col + i < n ? float(bias[col + i]) : 0.f));
        *reinterpret_cast<Pair*>(&out_tile[mi * 16 + g + ri * 8][warp * (TN / 4) + ni * 8 + t * 2]) = out;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int v = threadIdx.x; v < TM * (TN / 8); v += 128) {
    int row = v / (TN / 8), col = (v % (TN / 8)) * 8;
    int gr = blockIdx.x * TM + row, gc = blockIdx.y * TN + col;
    if (gr < m && gc + 7 < n) *reinterpret_cast<uint4*>(y + gr * n + gc) = *reinterpret_cast<const uint4*>(&out_tile[row][col]);
  }
}
thread_local std::string error_text;
struct Plan {
  int m, n;
  std::string info;
  Plan(int mm, int nn, int kk, void* b) : m(mm), n(nn) {
    if (kk != SHAPE_K || nn % 8 || (BIAS && !b)) throw std::runtime_error("shape/bias mismatch");
    std::ostringstream s;
    s << "{\"implementation\":\"CUTLASS small-K XOR-swizzled operand buffers\",\"tile\":[" << TM << ',' << TN << ',' << SHAPE_K
      << "],\"shared_bytes\":" << ((LOAD_MODE == 1 ? (TM + TN) * Stride * 2 : 0) + TM * (TN + 8) * 2) << ",\"grid\":["
      << (m + TM - 1) / TM << ',' << (n + TN - 1) / TN << ",1]}";
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
  small_k<<<dim3((q.m + TM - 1) / TM, (q.n + TN - 1) / TN), 128, 0, static_cast<cudaStream_t>(stream)>>>(
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
