#pragma once

#include <cstdint>
#include <stdexcept>

namespace phyai_sm110 {

inline void* Pointer(int64_t address) { return reinterpret_cast<void*>(address); }

inline int64_t Create(int64_t m, int64_t n, int64_t k, int64_t x, int64_t w, int64_t b, int64_t y, int64_t stream,
                      int64_t swizzle, int64_t raster) {
  void* plan;
#ifdef PHYAI_SCHEDULER_RUNTIME
  plan = ct_create_ex(m, n, k, Pointer(x), Pointer(w), Pointer(b), Pointer(y), Pointer(stream), swizzle, raster);
#else
  plan = ct_create(m, n, k, Pointer(x), Pointer(w), Pointer(b), Pointer(y), Pointer(stream));
#endif
  if (!plan) throw std::runtime_error(ct_error());
  return reinterpret_cast<int64_t>(plan);
}

inline void Run(int64_t plan, int64_t x, int64_t w, int64_t b, int64_t y, int64_t stream) {
  if (ct_run(Pointer(plan), Pointer(x), Pointer(w), Pointer(b), Pointer(y), Pointer(stream))) {
    throw std::runtime_error(ct_error());
  }
}

inline void Destroy(int64_t plan) { ct_destroy(Pointer(plan)); }

}  // namespace phyai_sm110
