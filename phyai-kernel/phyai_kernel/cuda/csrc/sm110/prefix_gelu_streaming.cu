#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/activation.cuh>

__device__ __forceinline__ float pi05_streaming_exact_gelu(const float& value) {
  constexpr float alpha = M_SQRT1_2;
  return value * 0.5f * (1.0f + ::erf(value * alpha));
}

template<int Threads>
__global__ __launch_bounds__(Threads) void pi05_prefix_streaming_gelu_kernel(__nv_bfloat16* __restrict__ output,
                                                                             const __nv_bfloat16* __restrict__ input,
                                                                             int dimension, int tile_vectors) {
  constexpr int vector_size = 8;
  const int64_t token = blockIdx.y;
  const int64_t input_offset = token * 2 * dimension;
  const int64_t output_offset = token * dimension;
  const int begin_vector = blockIdx.x * tile_vectors;
  const int end_vector = min(begin_vector + tile_vectors, dimension / vector_size);
  asm volatile("griddepcontrol.wait;");
  for (int vector = begin_vector + threadIdx.x; vector < end_vector; vector += Threads) {
    flashinfer::vec_t<float, vector_size> gate, up, activated;
    gate.cast_load(input + input_offset + vector * vector_size);
    up.cast_load(input + input_offset + dimension + vector * vector_size);
#pragma unroll
    for (int element = 0; element < vector_size; ++element) {
      activated[element] = pi05_streaming_exact_gelu(gate[element]) * up[element];
    }
    activated.cast_store(output + output_offset + vector * vector_size);
  }
  asm volatile("griddepcontrol.launch_dependents;");
}

template<int Threads>
cudaError_t launch_tiled(__nv_bfloat16* output, const __nv_bfloat16* input, int64_t tokens, int dimension, int tile,
                         bool enable_pdl, cudaStream_t stream) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = enable_pdl;
  cudaLaunchConfig_t configuration{};
  configuration.gridDim = dim3((dimension + tile - 1) / tile, tokens, 1);
  configuration.blockDim = dim3(Threads, 1, 1);
  configuration.dynamicSmemBytes = 0;
  configuration.stream = stream;
  configuration.attrs = &attribute;
  configuration.numAttrs = 1;
  return cudaLaunchKernelEx(&configuration, pi05_prefix_streaming_gelu_kernel<Threads>, output, input, dimension, tile / 8);
}

extern "C" int pi05_prefix_streaming_gelu(void* output, const void* input, int64_t tokens, int dimension, int threads, int tile,
                                          int enable_pdl, int device, uintptr_t stream_address) {
  if (tokens < 1 || tokens > 65535 || dimension < 8 || dimension % 8 || tile < 8 || tile % 8) { return cudaErrorInvalidValue; }
  if (output == nullptr || input == nullptr || output == input || threads != 128 || tile != 1024 || dimension > 65536) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) { return status; }
  auto destination = static_cast<__nv_bfloat16*>(output);
  auto source = static_cast<const __nv_bfloat16*>(input);
  auto stream = reinterpret_cast<cudaStream_t>(stream_address);
  switch (threads) {
    case 128: return launch_tiled<128>(destination, source, tokens, dimension, tile, enable_pdl, stream);
    case 256: return launch_tiled<256>(destination, source, tokens, dimension, tile, enable_pdl, stream);
    case 512: return launch_tiled<512>(destination, source, tokens, dimension, tile, enable_pdl, stream);
    case 1024: return launch_tiled<1024>(destination, source, tokens, dimension, tile, enable_pdl, stream);
    default: return cudaErrorInvalidValue;
  }
}
