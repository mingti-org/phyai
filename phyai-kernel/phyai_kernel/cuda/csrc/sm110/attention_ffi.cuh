#pragma once

#include <cstdint>
#include <stdexcept>

namespace phyai_sm110_attention {

inline void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline void Prepare() { Check(static_cast<cudaError_t>(cattn01_prepare())); }

inline void Run(int64_t q, int64_t k, int64_t v, int64_t indptr, int64_t indices, int64_t output, int64_t partial, int64_t lse,
                int64_t rows, int64_t q_stride, int64_t head_stride, int64_t k_stride, int64_t v_stride, double scale,
                int64_t stream) {
  Check(static_cast<cudaError_t>(cattn01_launch(
      reinterpret_cast<const void*>(q), reinterpret_cast<const void*>(k), reinterpret_cast<const void*>(v),
      reinterpret_cast<const int32_t*>(indptr), reinterpret_cast<const int32_t*>(indices), reinterpret_cast<void*>(output),
      reinterpret_cast<void*>(partial), reinterpret_cast<void*>(lse), rows, q_stride, head_stride, k_stride, v_stride, scale,
      PHYAI_TILE_ROWS, 128, reinterpret_cast<cudaStream_t>(stream))));
}

}  // namespace phyai_sm110_attention
