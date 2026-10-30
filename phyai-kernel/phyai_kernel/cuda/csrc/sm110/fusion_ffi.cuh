#pragma once

#include <cstdint>
#include <stdexcept>

namespace phyai_sm110_fusion {

inline void CreateTable(int64_t table, int64_t device, int64_t stream) {
  auto status = static_cast<cudaError_t>(pi05_gated_create_table(reinterpret_cast<void*>(table), device, stream));
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace phyai_sm110_fusion
