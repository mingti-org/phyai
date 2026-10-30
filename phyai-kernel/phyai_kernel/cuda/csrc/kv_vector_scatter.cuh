#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <cstdint>
#include <stdexcept>
#include <initializer_list>

namespace phyai_kv_vector {
const char* Data(const DLTensor* t) { return static_cast<const char*>(t->data) + t->byte_offset; }
// Each warp copies a complete 256-element row using sixteen-byte operations.
// Two warps per row keep K and V accesses independent and coalesced.
__global__ void Scatter(const uint4* k, const uint4* v, uint4* kc, uint4* vc, const int64_t* slots, int rows, int64_t kp,
                        int64_t vp, int capacity) {
  const int row = blockIdx.x * 2 + threadIdx.x / 64;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const bool value = (threadIdx.x & 63) >= 32;
  const int64_t slot = slots[row];
  // Slot zero is a write-only padding sentinel in the admitted production scope.
  // Never race duplicate padding writes. Positive indices must be unique and valid.
  if (slot <= 0 || slot >= capacity) return;
  if (value)
    vc[slot * 32 + lane] = v[row * vp + lane];
  else
    kc[slot * 32 + lane] = k[row * kp + lane];
}
}  // namespace phyai_kv_vector

void kv_vector_scatter(DLTensor* k, DLTensor* v, DLTensor* kc, DLTensor* vc, DLTensor* slots, int64_t stream) {
  using namespace phyai_kv_vector;
  if (!k || k->ndim != 3 || k->shape[0] < 1 || k->shape[0] > 4096 || k->shape[1] != 1 || k->shape[2] != 256)
    throw std::runtime_error("unsupported K/V row topology");
  const int rows = k->shape[0];
  for (auto* t : {k, v, kc, vc}) {
    if (!t || !t->data || t->device.device_type != kDLCUDA || t->device.device_id != k->device.device_id
        || t->dtype.code != kDLBfloat || t->dtype.bits != 16 || t->dtype.lanes != 1
        || reinterpret_cast<uintptr_t>(Data(t)) % 16)
      throw std::runtime_error("unsupported K/V storage");
  }
  for (auto* t : {k, v}) {
    if (t->ndim != 3 || t->shape[0] != rows || t->shape[1] != 1 || t->shape[2] != 256
        || (t->strides && (t->strides[0] < 256 || t->strides[0] % 8 || t->strides[1] != 256 || t->strides[2] != 1)))
      throw std::runtime_error("unsupported K/V source pitch");
  }
  for (auto* t : {kc, vc}) {
    if (t->ndim != 4 || t->shape[0] != 1019 || t->shape[1] != 1 || t->shape[2] != 1 || t->shape[3] != 256
        || (t->strides && (t->strides[0] != 256 || t->strides[3] != 1)))
      throw std::runtime_error("unsupported cache topology");
  }
  if (!slots || !slots->data || slots->ndim != 1 || slots->shape[0] != rows || slots->dtype.code != kDLInt
      || slots->dtype.bits != 64 || slots->dtype.lanes != 1 || slots->device.device_type != kDLCUDA
      || slots->device.device_id != k->device.device_id || (slots->strides && slots->strides[0] != 1)
      || reinterpret_cast<uintptr_t>(Data(slots)) % 8)
    throw std::runtime_error("unsupported slot metadata");
  const int64_t kp = (k->strides ? k->strides[0] : 256) / 8;
  const int64_t vp = (v->strides ? v->strides[0] : 256) / 8;
  Scatter<<<(rows + 1) / 2, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint4*>(Data(k)), reinterpret_cast<const uint4*>(Data(v)),
      reinterpret_cast<uint4*>(const_cast<char*>(Data(kc))), reinterpret_cast<uint4*>(const_cast<char*>(Data(vc))),
      reinterpret_cast<const int64_t*>(Data(slots)), rows, kp, vp, 1019);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
