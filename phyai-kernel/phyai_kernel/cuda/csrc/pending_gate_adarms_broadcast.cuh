#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace {
template<bool Packed>
__device__ __forceinline__ void Load8(const __nv_bfloat16* p, __nv_bfloat16* v) {
  if constexpr (Packed) {
    const uint4 q = *reinterpret_cast<const uint4*>(p);
    const unsigned int words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __ushort_as_bfloat16(static_cast<unsigned short>(words[j / 2] >> (16 * (j & 1))));
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = p[j];
  }
}
template<bool Packed>
__device__ __forceinline__ void Store8(__nv_bfloat16* p, const __nv_bfloat16* v) {
  if constexpr (Packed) {
    unsigned int words[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      words[j] = static_cast<unsigned int>(__bfloat16_as_ushort(v[2 * j]))
                 | (static_cast<unsigned int>(__bfloat16_as_ushort(v[2 * j + 1])) << 16);
    *reinterpret_cast<uint4*>(p) = make_uint4(words[0], words[1], words[2], words[3]);
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) p[j] = v[j];
  }
}

template<bool Packed, bool Broadcast>
__global__ void PendingGateAdaRms(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* gate,
                                  const __nv_bfloat16* mod, __nv_bfloat16* h, __nv_bfloat16* y, int64_t sa, int64_t sb,
                                  int64_t sg, int64_t sm, int64_t gm, int64_t mm, int64_t rows, float eps) {
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ float sums[4];
  __nv_bfloat16 av[8], bv[8], gv[8], hv[8];
  Load8<Packed>(a + row * sa + tid * 8, av);
  Load8<Packed>(b + row * sb + tid * 8, bv);
  Load8<Packed>(gate + (Broadcast ? 0 : (row / (rows / gm)) * sg) + tid * 8, gv);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    hv[j] = __float2bfloat16_rn(fmaf(__bfloat162float(bv[j]), __bfloat162float(gv[j]), __bfloat162float(av[j])));
  const float v0 = __bfloat162float(hv[0]), v1 = __bfloat162float(hv[1]);
  float sum = fmaf(v0, v0, __fmul_rn(v1, v1));
#pragma unroll
  for (int j = 2; j < 8; j += 2)
    sum = __fadd_rn(fmaf(__bfloat162float(hv[j]), __bfloat162float(hv[j]), sum),
                    __fmul_rn(__bfloat162float(hv[j + 1]), __bfloat162float(hv[j + 1])));
#pragma unroll
  for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
  if (lane == 0) sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 4 ? sums[lane] : 0.f;
#pragma unroll
    for (int off = 2; off; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) sums[0] = sum;
  }
  __syncthreads();
  const float variance = __fadd_rn(__fmul_rn(sums[0], 1.f / 1024.f), eps);
  float inv;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(variance));
  const int64_t mr = Broadcast ? 0 : row / (rows / mm);
  __nv_bfloat16 scales[8], shifts[8], yv[8];
  Load8<Packed>(mod + mr * sm + tid * 8, scales);
  Load8<Packed>(mod + mr * sm + 1024 + tid * 8, shifts);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    yv[j] = __float2bfloat16_rn(fmaf(__fmul_rn(__bfloat162float(hv[j]), inv), __fadd_rn(1.f, __bfloat162float(scales[j])),
                                     __bfloat162float(shifts[j])));
  Store8<Packed>(h + row * 1024 + tid * 8, hv);
  Store8<Packed>(y + row * 1024 + tid * 8, yv);
}
const __nv_bfloat16* Ptr(const DLTensor* t) {
  return reinterpret_cast<const __nv_bfloat16*>(static_cast<const char*>(t->data) + t->byte_offset);
}
}  // namespace
void pending_gate_adarms(DLTensor* a, DLTensor* b, DLTensor* g, DLTensor* m, DLTensor* h, DLTensor* y, double epsilon,
                         int64_t stream) {
  if (!a || a->ndim != 2 || a->shape[0] <= 0 || a->shape[1] != 1024) throw std::runtime_error("invalid expert input");
  const int64_t rows = a->shape[0];
  for (auto* t : {a, b, g, m, h, y}) {
    if (!t || !t->data || t->ndim != 2 || t->device.device_type != kDLCUDA || t->device.device_id != a->device.device_id
        || t->dtype.code != kDLBfloat || t->dtype.bits != 16 || t->dtype.lanes != 1 || reinterpret_cast<uintptr_t>(Ptr(t)) % 2
        || t->shape[0] <= 0 || rows % t->shape[0] || t->shape[1] != (t == m ? 3072 : 1024)
        || (t->strides && (t->strides[1] != 1 || t->strides[0] < t->shape[1])))
      throw std::runtime_error("invalid expert tensor layout");
  }
  if (b->shape[0] != rows || h->shape[0] != rows || y->shape[0] != rows || (h->strides && h->strides[0] != 1024)
      || (y->strides && y->strides[0] != 1024))
    throw std::runtime_error("invalid output layout");
  const int64_t sa = a->strides ? a->strides[0] : 1024, sb = b->strides ? b->strides[0] : 1024,
                sg = g->strides ? g->strides[0] : 1024, sm = m->strides ? m->strides[0] : 3072;
  bool packed = sa % 8 == 0 && sb % 8 == 0 && sg % 8 == 0 && sm % 8 == 0;
  for (auto* t : {a, b, g, m, h, y}) packed = packed && reinterpret_cast<uintptr_t>(Ptr(t)) % 16 == 0;
  auto st = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
  const bool broadcast = g->shape[0] == 1 && m->shape[0] == 1;
  if (broadcast) {
    if (packed)
      PendingGateAdaRms<true, true><<<rows, 128, 0, st>>>(Ptr(a), Ptr(b), Ptr(g), Ptr(m), const_cast<__nv_bfloat16*>(Ptr(h)),
                                                          const_cast<__nv_bfloat16*>(Ptr(y)), sa, sb, sg, sm, g->shape[0],
                                                          m->shape[0], rows, static_cast<float>(epsilon));
    else
      PendingGateAdaRms<false, true><<<rows, 128, 0, st>>>(Ptr(a), Ptr(b), Ptr(g), Ptr(m), const_cast<__nv_bfloat16*>(Ptr(h)),
                                                           const_cast<__nv_bfloat16*>(Ptr(y)), sa, sb, sg, sm, g->shape[0],
                                                           m->shape[0], rows, static_cast<float>(epsilon));
  } else {
    if (packed)
      PendingGateAdaRms<true, false><<<rows, 128, 0, st>>>(Ptr(a), Ptr(b), Ptr(g), Ptr(m), const_cast<__nv_bfloat16*>(Ptr(h)),
                                                           const_cast<__nv_bfloat16*>(Ptr(y)), sa, sb, sg, sm, g->shape[0],
                                                           m->shape[0], rows, static_cast<float>(epsilon));
    else
      PendingGateAdaRms<false, false><<<rows, 128, 0, st>>>(Ptr(a), Ptr(b), Ptr(g), Ptr(m), const_cast<__nv_bfloat16*>(Ptr(h)),
                                                            const_cast<__nv_bfloat16*>(Ptr(y)), sa, sb, sg, sm, g->shape[0],
                                                            m->shape[0], rows, static_cast<float>(epsilon));
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
