#include "radix_cache/atoms.h"

// Use the upstream xxHash library in header-only mode. XXH_INLINE_ALL makes all
// symbols static-inline (namespaced to XXH_INLINE_) so there are no ODR issues
// and no separate translation unit is required — this file stays the sole TU
// that pulls in xxhash.h, and the CMake glob (*.cpp/*.cc/*.cxx) is unaffected.
// The header lives at phyai-ext/third_party/xxhash/ and is surfaced onto the
// include path by CMakeLists.txt.
#define XXH_INLINE_ALL
#include "xxhash.h"

namespace phyai_ext::radix_cache {

std::uint64_t xxh3_64(const void* data, std::size_t bytes) noexcept {
  return static_cast<std::uint64_t>(XXH3_64bits(data, bytes));
}

std::size_t atom_vec_hash::operator()(const atom_vec& v) const noexcept {
  return static_cast<std::size_t>(xxh3_64(v.data(), v.size()));
}

}  // namespace phyai_ext::radix_cache
