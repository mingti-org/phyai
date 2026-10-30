#include "prefix_gelu_streaming.cu"

__global__ void pi05_gated_table_kernel(float* table) {
  unsigned pattern = blockIdx.x * blockDim.x + threadIdx.x;
  if (pattern < 65536) { table[pattern] = pi05_streaming_exact_gelu(__bfloat162float(__ushort_as_bfloat16(pattern))); }
}

extern "C" int pi05_gated_create_table(void* table, int device, uintptr_t stream_address) {
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) return status;
  pi05_gated_table_kernel<<<256, 256, 0, reinterpret_cast<cudaStream_t>(stream_address)>>>(static_cast<float*>(table));
  return cudaGetLastError();
}

#include "fusion_ffi.cuh"
