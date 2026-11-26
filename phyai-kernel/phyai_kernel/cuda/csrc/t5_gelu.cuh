#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/memory.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace phyai_kernel {
namespace t5_gelu_detail {

// PyTorch's pow(x, 3) rounds both multiplies in x's dtype. The scalar
// coefficients, the sum, tanh, and the final products also round separately.
__device__ __forceinline__ __half2 activate(__half2 x) {
  float2 cube = __half22float2(__hmul2(__hmul2(x, x), x));
  __half2 polynomial = __hadd2(x, __floats2half2_rn(0.044715f * cube.x, 0.044715f * cube.y));
  float2 p = __half22float2(polynomial);
  __half2 argument = __floats2half2_rn(0.7978845608028654f * p.x, 0.7978845608028654f * p.y);
  __half2 factor = __hadd2(__float2half2_rn(1.0f), h2tanh(argument));
  return __hmul2(__hmul2(__float2half2_rn(0.5f), x), factor);
}

__device__ __forceinline__ __nv_bfloat162 activate(__nv_bfloat162 x) {
  __nv_bfloat162 cube = __hmul2(__hmul2(x, x), x);
  // Split each FP32 coefficient into two BF16 parts. Packed FMA avoids
  // scalar conversions; all BF16 inputs retain the eager expression's result.
  __nv_bfloat162 cubic =
      __hfma2(cube, __float2bfloat162_rn(0.044677734375f), __hmul2(cube, __float2bfloat162_rn(0.044715f - 0.044677734375f)));
  __nv_bfloat162 polynomial = __hadd2(x, cubic);
  __nv_bfloat162 argument = __hfma2(polynomial, __float2bfloat162_rn(0.796875f),
                                    __hmul2(polynomial, __float2bfloat162_rn(0.7978845608028654f - 0.796875f)));
  // h2tanh retains BF16 rounding through the FP32 tanh instruction.
  // The lower-accuracy h2tanh_approx instruction changes valid BF16 results.
  __nv_bfloat162 factor = __hadd2(__float2bfloat162_rn(1.0f), h2tanh(argument));
  return __hmul2(__hmul2(__float2bfloat162_rn(0.5f), x), factor);
}

template<typename T>
using Packed = std::conditional_t<std::is_same_v<T, cutlass::half_t>, __half2, __nv_bfloat162>;

template<typename T, int threads, int vector_size, bool contiguous, bool flat, bool full_tiles, typename Index>
__global__ void vector_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t count, int columns,
                              int64_t row_stride, cutlass::FastDivmod divider) {
  using namespace cute;
  constexpr int copies = vector_size / 8;
  constexpr int tile_size = threads * vector_size;
  int row = blockIdx.y;
  Index length = contiguous || flat ? Index(count) : Index(columns);
  Index start = Index(blockIdx.x) * tile_size + threadIdx.x * 8;
  if constexpr (!full_tiles) {
    if (start >= length) return;
  }
  auto values = make_tensor<T>(make_layout(make_shape(_8{}, Int<copies>{})));
  if constexpr (!full_tiles) clear(values);
  CUTE_UNROLL
  for (int i = 0; i < copies; ++i) {
    Index index = start + i * threads * 8;
    bool valid = full_tiles || index < length;
    int64_t source_index = index;
    if constexpr (!contiguous) {
      if constexpr (flat) {
        int input_row, input_column;
        divider(input_row, input_column, int(index));
        source_index = int64_t(input_row) * row_stride + input_column;
      } else {
        source_index += int64_t(row) * row_stride;
      }
    }
    if (valid) {
      auto source = make_tensor(make_gmem_ptr(input + source_index), make_layout(_8{}));
      copy(Copy_Atom<UniversalCopy<uint128_t>, T>{}, source, values(_, i));
    }
  }
  auto pairs = recast<Packed<T>>(values);
  CUTE_UNROLL
  for (int i = 0; i < size(pairs); ++i) pairs(i) = activate(pairs(i));
  auto vectors = recast<uint4>(values);
  CUTE_UNROLL
  for (int i = 0; i < copies; ++i) {
    Index index = start + i * threads * 8;
    int64_t destination_index = contiguous || flat ? index : int64_t(row) * columns + index;
    // Keep a 128-bit transaction and avoid displacing input cache lines with
    // streaming output. A plain uint128_t assignment can split into 64-bit stores.
    if (full_tiles || index < length) __stcs(reinterpret_cast<uint4*>(output + destination_index), vectors(i));
  }
}

constexpr int MAX_RANK = 64;

struct ViewShape {
  int rank = 0;
  int64_t sizes[MAX_RANK]{};
  int64_t strides[MAX_RANK]{};
};

// Minor dimensions come first. Collapse adjacent dimensions only when their
// actual strides agree, including singleton and broadcast dimensions.
inline ViewShape view_shape(tvm::ffi::TensorView input) {
  ViewShape shape;
  for (int axis = input.ndim() - 1; axis >= 0; --axis) {
    int64_t size = input.size(axis);
    int64_t stride = input.stride(axis);
    TVM_FFI_ICHECK_GE(stride, 0) << "t5_gelu requires nonnegative strides";
    if (size == 1) continue;
    int previous = shape.rank - 1;
    if (previous >= 0 && shape.strides[previous] * shape.sizes[previous] == stride) {
      shape.sizes[previous] *= size;
    } else {
      shape.sizes[shape.rank] = size;
      shape.strides[shape.rank++] = stride;
    }
  }
  if (shape.rank == 0) {
    shape.rank = 1;
    shape.sizes[0] = 1;
  }
  return shape;
}

template<typename Divmod, int rank_capacity>
struct ViewLayout {
  static constexpr int capacity = rank_capacity == 0 ? MAX_RANK : rank_capacity;
  int rank;
  int64_t strides[capacity];
  Divmod dividers[capacity];

  explicit ViewLayout(const ViewShape& shape) : rank(shape.rank), strides{}, dividers{} {
    for (int i = 0; i < rank; ++i) {
      strides[i] = shape.strides[i];
      if (i + 1 < rank) dividers[i] = Divmod(shape.sizes[i]);
    }
  }

  template<typename Index>
  __device__ __forceinline__ int64_t offset(Index index) const {
    int64_t result = 0;
    int dimensions = rank_capacity == 0 ? rank : rank_capacity;
    CUTE_UNROLL
    for (int i = 0; i + 1 < dimensions; ++i) {
      Index quotient, remainder;
      dividers[i](quotient, remainder, index);
      result += int64_t(remainder) * strides[i];
      index = quotient;
    }
    return result + int64_t(index) * strides[dimensions - 1];
  }
};

template<typename T, typename Index, typename Divmod, int rank_capacity, int pack = 1, bool contiguous = false>
__global__ void strided_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t count,
                               const __grid_constant__ ViewLayout<Divmod, rank_capacity> layout) {
  using namespace cute;
  constexpr int threads = 128;
  constexpr int vector_size = 8;
  Index start = Index(blockIdx.x) * threads * vector_size + threadIdx.x * pack;
  if (start >= count) return;
  auto values = make_tensor<T>(make_layout(Int<vector_size>{}));
  // Use scalar loads for general views. Aligned output still permits packed
  // stores when the input pointer or stride prevents vector loads.
  CUTE_UNROLL
  for (int i = 0; i < vector_size; ++i) {
    Index index = start + (i / pack) * threads * pack + i % pack;
    int64_t source_index = contiguous ? int64_t(index) : layout.offset(index);
    values(i) = index < count ? input[source_index] : T(0.0f);
  }
  auto pairs = recast<Packed<T>>(values);
  CUTE_UNROLL
  for (int i = 0; i < size(pairs); ++i) pairs(i) = activate(pairs(i));
  if constexpr (pack == 2) {
    auto words = recast<uint32_t>(values);
    CUTE_UNROLL
    for (int i = 0; i < vector_size / 2; ++i) {
      Index index = start + i * threads * 2;
      if (index + 1 < count) {
        cutlass::arch::global_store<uint32_t, 4>(words(i), output + index, true);
      } else if (index < count) {
        output[index] = values(i * 2);
      }
    }
  } else {
    CUTE_UNROLL
    for (int i = 0; i < vector_size; ++i) {
      Index index = start + i * threads;
      if (index < count) output[index] = values(i);
    }
  }
}

template<typename T, typename Index, typename Divmod, int rank_capacity, int pack = 1, bool contiguous = false>
void launch_strided(const T* input, T* output, int64_t count, const ViewShape& shape, cudaStream_t stream) {
  unsigned blocks = (count + 1023) / 1024;
  strided_kernel<T, Index, Divmod, rank_capacity, pack, contiguous>
      <<<blocks, 128, 0, stream>>>(input, output, count, ViewLayout<Divmod, rank_capacity>(shape));
}

template<typename T, typename Index, typename Divmod>
void dispatch_strided(const T* input, T* output, int64_t count, const ViewShape& shape, cudaStream_t stream) {
  switch (shape.rank) {
    case 1:
      if (shape.strides[0] == 1 && (uintptr_t(output) & 3) == 0) {
        launch_strided<T, Index, Divmod, 1, 2, true>(input, output, count, shape, stream);
      } else if ((uintptr_t(output) & 3) == 0) {
        launch_strided<T, Index, Divmod, 1, 2>(input, output, count, shape, stream);
      } else {
        launch_strided<T, Index, Divmod, 1>(input, output, count, shape, stream);
      }
      break;
    case 2: launch_strided<T, Index, Divmod, 2>(input, output, count, shape, stream); break;
    case 3: launch_strided<T, Index, Divmod, 3>(input, output, count, shape, stream); break;
    default: launch_strided<T, Index, Divmod, 0>(input, output, count, shape, stream);
  }
}

template<typename T, bool vectorized, bool flat>
__global__ void transpose_kernel(const T* __restrict__ input, T* __restrict__ output, int rows, int columns,
                                 int64_t column_stride, cutlass::FastDivmod tiles_across) {
  using namespace cute;
  constexpr int tile_rows = 64;
  constexpr int tile_columns = std::is_same_v<T, cutlass::bfloat16_t> ? 64 : 32;
  constexpr int threads = tile_rows * tile_columns / 16;
  // Each shared word holds two rows. Padding separates column banks while
  // preserving packed loads; the output gathers eight columns per row pair.
  __shared__ uint32_t storage[(tile_rows / 2 + 1) * tile_columns];
  auto tile = make_tensor(make_smem_ptr(storage), make_layout(make_shape(Int<tile_rows / 2>{}, Int<tile_columns>{}),
                                                              make_stride(_1{}, Int<tile_rows / 2 + 1>{})));
  int tile_row = blockIdx.y;
  int tile_column = blockIdx.x;
  if constexpr (flat) tiles_across(tile_row, tile_column, int(blockIdx.x));
  auto values = make_tensor<T>(make_layout(make_shape(_8{}, _2{})));
  clear(values);
  CUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    int offset = threadIdx.x * 8 + i * threads * 8;
    int row = tile_row * tile_rows + offset % tile_rows;
    int column = tile_column * tile_columns + offset / tile_rows;
    if constexpr (vectorized) {
      if (row < rows && column < columns) {
        auto source = make_tensor(make_gmem_ptr(input + row + int64_t(column) * column_stride), make_layout(_8{}));
        copy(Copy_Atom<UniversalCopy<uint128_t>, T>{}, source, values(_, i));
      }
    } else {
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j) {
        if (row + j < rows && column < columns) values(j, i) = input[row + j + int64_t(column) * column_stride];
      }
    }
  }
  auto pairs = recast<Packed<T>>(values);
  CUTE_UNROLL
  for (int i = 0; i < size(pairs); ++i) pairs(i) = activate(pairs(i));
  auto words = recast<uint32_t>(values);
  CUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    int offset = threadIdx.x * 8 + i * threads * 8;
    CUTE_UNROLL
    for (int j = 0; j < 4; ++j) tile(offset % tile_rows / 2 + j, offset / tile_rows) = words(j, i);
  }
  __syncthreads();
  int row = threadIdx.x / (tile_columns / 8);
  int column = threadIdx.x % (tile_columns / 8) * 8;
  auto packed = make_tensor<uint32_t>(make_layout(_8{}));
  CUTE_UNROLL
  for (int j = 0; j < 8; ++j) packed(j) = tile(row, column + j);
  auto halves = recast<T>(packed);
  auto result = make_tensor<T>(make_layout(make_shape(_8{}, _2{})));
  CUTE_UNROLL
  for (int j = 0; j < 8; ++j) {
    result(j, 0) = halves(j * 2);
    result(j, 1) = halves(j * 2 + 1);
  }
  auto vectors = recast<uint4>(result);
  int output_row = tile_row * tile_rows + row * 2;
  int output_column = tile_column * tile_columns + column;
  CUTE_UNROLL
  for (int i = 0; i < 2; ++i) {
    int64_t index = int64_t(output_row + i) * columns + output_column;
    if constexpr (vectorized) {
      cutlass::arch::global_store<uint4, 16>(vectors(0, i), output + index, output_row + i < rows && output_column < columns);
    } else {
      CUTE_UNROLL
      for (int j = 0; j < 8; ++j) {
        if (output_row + i < rows && output_column + j < columns) output[index + j] = result(j, i);
      }
    }
  }
}

template<typename T, bool vectorized>
void launch_transpose(const T* input, T* output, int rows, int columns, int64_t stride, cudaStream_t stream) {
  constexpr int tile_columns = std::is_same_v<T, cutlass::bfloat16_t> ? 64 : 32;
  constexpr int threads = tile_columns * 4;
  int tiles_across = (int64_t(columns) + tile_columns - 1) / tile_columns;
  int tiles_down = (int64_t(rows) + 63) / 64;
  auto divider = cutlass::FastDivmod(tiles_across);
  if (tiles_down <= 65535) {
    transpose_kernel<T, vectorized, false>
        <<<dim3(tiles_across, tiles_down), threads, 0, stream>>>(input, output, rows, columns, stride, divider);
  } else {
    transpose_kernel<T, vectorized, true>
        <<<tiles_across * tiles_down, threads, 0, stream>>>(input, output, rows, columns, stride, divider);
  }
}

template<typename T, int threads, int vector_size, bool contiguous, bool flat, typename Index = uint32_t>
void launch_vector(const T* input, T* output, int64_t count, int columns, int rows, int64_t stride, cudaStream_t stream) {
  if constexpr (!contiguous && !flat) {
    if (rows > 65535) {
      launch_vector<T, threads, vector_size, false, true, Index>(input, output, count, columns, rows, stride, stream);
      return;
    }
  }
  constexpr int tile_size = threads * vector_size;
  int64_t length = contiguous || flat ? count : columns;
  dim3 grid((length + tile_size - 1) / tile_size, contiguous || flat ? 1 : rows);
  auto divider = cutlass::FastDivmod(contiguous ? 1 : columns);
  if (length % tile_size == 0) {
    vector_kernel<T, threads, vector_size, contiguous, flat, true, Index>
        <<<grid, threads, 0, stream>>>(input, output, count, columns, stride, divider);
  } else {
    vector_kernel<T, threads, vector_size, contiguous, flat, false, Index>
        <<<grid, threads, 0, stream>>>(input, output, count, columns, stride, divider);
  }
}

template<typename T>
void launch(tvm::ffi::TensorView input, tvm::ffi::TensorView output, const ViewShape& shape, cudaStream_t stream) {
  const auto* source = reinterpret_cast<const T*>(static_cast<const char*>(input.data_ptr()) + input.byte_offset());
  auto* destination = reinterpret_cast<T*>(static_cast<char*>(output.data_ptr()) + output.byte_offset());
  int64_t count = input.numel();
  bool aligned = ((uintptr_t(source) | uintptr_t(destination)) & 15) == 0;
  if (aligned && shape.rank == 1 && shape.strides[0] == 1 && count % 8 == 0) {
    if (count <= std::numeric_limits<int>::max()) {
      launch_vector<T, 128, 16, true, true>(source, destination, count, 1, 1, 0, stream);
    } else {
      launch_vector<T, 128, 16, true, true, int64_t>(source, destination, count, 1, 1, 0, stream);
    }
  } else if (aligned && shape.rank == 2 && shape.strides[0] == 1 && shape.sizes[0] % 8 == 0 && shape.strides[1] % 8 == 0
             && count <= std::numeric_limits<int>::max()) {
    int columns = shape.sizes[0];
    int rows = shape.sizes[1];
    if (std::is_same_v<T, cutlass::bfloat16_t> && columns <= 2048 && count <= (1 << 20)) {
      launch_vector<T, 256, 8, false, false>(source, destination, count, columns, rows, shape.strides[1], stream);
    } else if (columns <= 4096) {
      launch_vector<T, 128, 8, false, true>(source, destination, count, columns, rows, shape.strides[1], stream);
    } else if (columns % 2048 == 0) {
      launch_vector<T, 128, 16, false, false>(source, destination, count, columns, rows, shape.strides[1], stream);
    } else if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
      launch_vector<T, 128, 16, false, true>(source, destination, count, columns, rows, shape.strides[1], stream);
    } else {
      launch_vector<T, 128, 8, false, false>(source, destination, count, columns, rows, shape.strides[1], stream);
    }
  } else if (shape.rank == 2 && shape.strides[1] == 1 && count <= std::numeric_limits<int>::max()) {
    int columns = shape.sizes[0];
    int rows = shape.sizes[1];
    if (aligned && rows % 8 == 0 && columns % 8 == 0 && shape.strides[0] % 8 == 0) {
      launch_transpose<T, true>(source, destination, rows, columns, shape.strides[0], stream);
    } else {
      launch_transpose<T, false>(source, destination, rows, columns, shape.strides[0], stream);
    }
  } else {
    if (count <= std::numeric_limits<int>::max()) {
      dispatch_strided<T, int, cutlass::FastDivmod>(source, destination, count, shape, stream);
    } else {
      dispatch_strided<T, uint64_t, cutlass::FastDivmodU64>(source, destination, count, shape, stream);
    }
  }
}

}  // namespace t5_gelu_detail

inline void t5_gelu(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
  TVM_FFI_ICHECK_EQ(input.device().device_type, kDLCUDA) << "t5_gelu requires CUDA tensors";
  TVM_FFI_ICHECK_EQ(output.device().device_type, kDLCUDA);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  TVM_FFI_ICHECK(input.dtype().bits == 16 && input.dtype().lanes == 1
                 && (input.dtype().code == kDLFloat || input.dtype().code == kDLBfloat))
      << "t5_gelu supports float16 and bfloat16";
  TVM_FFI_ICHECK_EQ(input.dtype().code, output.dtype().code);
  TVM_FFI_ICHECK_EQ(input.dtype().bits, output.dtype().bits);
  TVM_FFI_ICHECK_EQ(input.dtype().lanes, output.dtype().lanes);
  TVM_FFI_ICHECK_EQ(input.ndim(), output.ndim());
  TVM_FFI_ICHECK_LE(input.ndim(), t5_gelu_detail::MAX_RANK);
  for (int i = 0; i < input.ndim(); ++i) TVM_FFI_ICHECK_EQ(input.size(i), output.size(i));
  if (input.numel() == 0) return;
  TVM_FFI_ICHECK(output.IsContiguous()) << "t5_gelu output must be contiguous";

  auto shape = t5_gelu_detail::view_shape(input);
  uintptr_t source = uintptr_t(input.data_ptr()) + input.byte_offset();
  uintptr_t destination = uintptr_t(output.data_ptr()) + output.byte_offset();
  uint64_t source_bytes = 2;
  for (int i = 0; i < shape.rank; ++i) source_bytes += 2 * (shape.sizes[i] - 1) * shape.strides[i];
  uint64_t output_bytes = 2 * output.numel();
  TVM_FFI_ICHECK(source + source_bytes <= destination || destination + output_bytes <= source)
      << "t5_gelu input and output must not overlap";

  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, input.device().device_id));
  if (input.dtype().code == kDLBfloat) {
    t5_gelu_detail::launch<cutlass::bfloat16_t>(input, output, shape, stream);
  } else {
    t5_gelu_detail::launch<cutlass::half_t>(input, output, shape, stream);
  }
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace phyai_kernel
