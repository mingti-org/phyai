#pragma once

#include "cutlass/array.h"
#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"

template<typename Compute>
struct Bf16ExactGelu {
  static constexpr bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  Compute operator()(Compute const& value) const {
    auto rounded = cutlass::NumericConverter<cutlass::bfloat16_t, Compute>{}(value);
    return cutlass::epilogue::thread::GELU<Compute>{}(Compute(rounded));
  }
};

template<typename Compute, int Elements>
struct Bf16ExactGelu<cutlass::Array<Compute, Elements>> {
  static constexpr bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE
  cutlass::Array<Compute, Elements> operator()(cutlass::Array<Compute, Elements> const& values) const {
    cutlass::Array<Compute, Elements> result;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Elements; ++index) { result[index] = Bf16ExactGelu<Compute>{}(values[index]); }
    return result;
  }
};
