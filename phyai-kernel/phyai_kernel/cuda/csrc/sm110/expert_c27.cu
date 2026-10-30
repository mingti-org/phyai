#include <cuda_runtime.h>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/array.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/memory_sm80.h>
#include <cutlass/gemm/collective/builders/sm100_common.inl>

#ifndef CATTN_SPLITS
#define CATTN_SPLITS 8
#endif

using namespace cute;
using Scalar = cutlass::bfloat16_t;

__device__ __forceinline__ float sfu_exp2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

namespace Builder = cutlass::gemm::collective::detail;
using QueryKeyMma = decltype(Builder::sm100_make_1sm_trivial_tiled_mma<Scalar, Scalar, float, Shape<_64, _128, _256>,
                                                                       Shape<_1, _1, _1>, UMMA::Major::K, UMMA::Major::K>());
using ProbabilityValueMma =
    decltype(Builder::sm100_make_1sm_trivial_tiled_mma<Scalar, Scalar, float, Shape<_64, _256, _128>, Shape<_1, _1, _1>,
                                                       UMMA::Major::K, UMMA::Major::MN>());
using QueryAtom = decltype(Builder::sm100_smem_selector<UMMA::Major::K, Scalar, _64, _256>());
using KeyAtom = decltype(Builder::sm100_smem_selector<UMMA::Major::K, Scalar, _128, _256>());
using ValueAtom = decltype(Builder::sm100_smem_selector<UMMA::Major::MN, Scalar, _256, _128>());
using ProbabilityAtom = decltype(Builder::sm100_smem_selector<UMMA::Major::K, Scalar, _64, _128>());
using QueryLayout = decltype(tile_to_shape(QueryAtom{}, Shape<_64, _256>{}, Step<_2, _1>{}));
using KeyLayout = decltype(tile_to_shape(KeyAtom{}, Shape<_128, _256>{}, Step<_2, _1>{}));
using ValueLayout = decltype(tile_to_shape(ValueAtom{}, Shape<_256, _128>{}, Step<_1, _2>{}));
using ProbabilityLayout = decltype(tile_to_shape(ProbabilityAtom{}, Shape<_64, _128>{}, Step<_2, _1>{}));
static_assert(cosize_v<KeyLayout> == cosize_v<ValueLayout>);

struct alignas(128) AttentionStorage {
  cutlass::AlignedArray<Scalar, cosize_v<QueryLayout>, 128> query;
  cutlass::AlignedArray<Scalar, cosize_v<KeyLayout>, 128> key;
  cutlass::AlignedArray<Scalar, cosize_v<ProbabilityLayout>, 128> probability;
  alignas(16) uint64_t barrier;
  uint32_t tmem;
};

__global__ void attention_cutlass_tcgen_v20(const Scalar* query, const Scalar* keys, const Scalar* values,
                                            const int32_t* indptr, const int32_t* indices, Scalar* direct_output,
                                            float* partial, float* logsumexp, int query_rows, int64_t query_stride,
                                            int64_t head_stride, int64_t key_stride, int64_t value_stride, float scale) {
  extern __shared__ __align__(128) unsigned char shared_memory[];
  auto& storage = *reinterpret_cast<AttentionStorage*>(shared_memory);
  auto shared_query = make_tensor(make_smem_ptr(storage.query.data()), QueryLayout{});
  auto shared_key = make_tensor(make_smem_ptr(storage.key.data()), KeyLayout{});
  auto shared_value = make_tensor(make_smem_ptr(storage.key.data()), ValueLayout{});
  auto shared_probability = make_tensor(make_smem_ptr(storage.probability.data()), ProbabilityLayout{});
  int key_count = indptr[1] - indptr[0];
  int key_tiles = (key_count + 127) / 128;
  int first_tile = key_tiles * blockIdx.y / CATTN_SPLITS;
  int last_tile = key_tiles * (blockIdx.y + 1) / CATTN_SPLITS;
  int query_start = blockIdx.x * 64;
  if (threadIdx.x == 0) cutlass::arch::ClusterBarrier::init(&storage.barrier, 1);
  cutlass::arch::fence_barrier_init();
  TMEM::Allocator1Sm allocator;
  if (threadIdx.x < 32) {
    allocator.allocate(256, &storage.tmem);
    allocator.release_allocation_lock();
  }
  for (int vector = threadIdx.x; vector < 64 * 32; vector += 128) {
    int row = vector / 32;
    int dimension = (vector % 32) * 8;
    int flat_row = query_start + row;
    bool valid = flat_row < query_rows * 8;
    int source_row = valid ? flat_row : 0;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
        &shared_query(row, dimension), query + (source_row / 8) * query_stride + (source_row % 8) * head_stride + dimension,
        valid);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();
  QueryKeyMma query_key;
  ProbabilityValueMma probability_value;
  auto query_key_slice = query_key.get_slice(0);
  auto probability_value_slice = probability_value.get_slice(0);
  auto fragment_query = query_key_slice.make_fragment_A(query_key_slice.partition_A(shared_query));
  auto fragment_key = query_key_slice.make_fragment_B(query_key_slice.partition_B(shared_key));
  auto fragment_probability = probability_value_slice.make_fragment_A(probability_value_slice.partition_A(shared_probability));
  auto fragment_value = probability_value_slice.make_fragment_B(probability_value_slice.partition_B(shared_value));
  auto scores = partition_fragment_C(query_key, Shape<_64, _128>{});
  auto outputs = partition_fragment_C(probability_value, Shape<_64, _256>{});
  scores.data() = storage.tmem + (16 << 16);
  outputs.data() = storage.tmem;
  auto score_coordinates = query_key_slice.partition_C(make_identity_tensor(Shape<_64, _128>{}));
  auto output_coordinates = probability_value_slice.partition_C(make_identity_tensor(Shape<_64, _256>{}));
  auto score_copy = make_tmem_copy(SM100_TMEM_LOAD_16dp128b16x{}, scores);
  auto output_copy = make_tmem_copy(SM100_TMEM_LOAD_16dp128b8x{}, outputs);
  auto output_store = make_tmem_copy(SM100_TMEM_STORE_16dp128b8x{}, outputs);
  auto score_thread = score_copy.get_slice(threadIdx.x);
  auto output_thread = output_copy.get_slice(threadIdx.x);
  auto output_store_thread = output_store.get_slice(threadIdx.x);
  auto thread_scores = score_thread.partition_S(scores);
  auto thread_score_coordinates = score_thread.partition_D(score_coordinates);
  auto thread_outputs = output_thread.partition_S(outputs);
  auto thread_output_coordinates = output_thread.partition_D(output_coordinates);
  auto thread_output_store = output_store_thread.partition_D(outputs);
  float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float denominator[2] = {0.0f, 0.0f};
  int barrier_phase = 0;
  for (int tile = first_tile; tile < last_tile; ++tile) {
    int key_start = tile * 128;
    int page_lane = threadIdx.x % 32;
    int page_row = threadIdx.x / 32 + page_lane * 4;
    bool page_valid = key_start + page_row < key_count;
    int prefetched_page = page_valid ? indices[indptr[0] + key_start + page_row] : 0;
    for (int vector = threadIdx.x; vector < 128 * 32; vector += 128) {
      int row = vector / 32;
      int dimension = (vector % 32) * 8;
      bool valid = key_start + row < key_count;
      int page = __shfl_sync(0xffffffff, prefetched_page, row / 4);
      cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(&shared_key(row, dimension),
                                                                               keys + page * key_stride + dimension, valid);
    }
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if (threadIdx.x < 32) {
      query_key.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int reduction = 0; reduction < size<2>(fragment_query); ++reduction) {
        gemm(query_key, fragment_query(_, _, reduction), fragment_key(_, _, reduction), scores);
        query_key.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.barrier);
    }
    cutlass::arch::ClusterBarrier::wait(&storage.barrier, barrier_phase);
    barrier_phase ^= 1;
    for (int vector = threadIdx.x; vector < 128 * 32; vector += 128) {
      int row = vector / 32;
      int dimension = (vector % 32) * 8;
      bool valid = key_start + row < key_count;
      int page = __shfl_sync(0xffffffff, prefetched_page, row / 4);
      cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(&shared_value(dimension, row),
                                                                               values + page * value_stride + dimension, valid);
    }
    cutlass::arch::cp_async_fence();
    auto score_registers = make_tensor<float>(shape(thread_score_coordinates));
    copy(score_copy, thread_scores, score_registers);
    cutlass::arch::fence_view_async_tmem_load();
    float local_maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int element = 0; element < size(score_registers); ++element) {
      auto coordinate = thread_score_coordinates(element);
      float score = key_start + int(get<1>(coordinate)) < key_count ? score_registers(element) * scale * 1.4426950408889634f
                                                                    : -CUDART_INF_F;
      score_registers(element) = score;
      local_maximum[element % 2] = fmaxf(local_maximum[element % 2], score);
    }
    float new_maximum[2];
    float correction[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float reduced = local_maximum[half];
      reduced = fmaxf(reduced, __shfl_xor_sync(0xffffffff, reduced, 1));
      reduced = fmaxf(reduced, __shfl_xor_sync(0xffffffff, reduced, 2));
      new_maximum[half] = fmaxf(maximum[half], reduced);
      correction[half] = sfu_exp2(maximum[half] - new_maximum[half]);
      denominator[half] *= correction[half];
    }
    float local_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int element = 0; element < size(score_registers); ++element) {
      auto coordinate = thread_score_coordinates(element);
      float probability = sfu_exp2(score_registers(element) - new_maximum[element % 2]);
      local_sum[element % 2] += probability;
      shared_probability(get<0>(coordinate), get<1>(coordinate)) = Scalar(probability);
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float reduced = local_sum[half];
      reduced += __shfl_xor_sync(0xffffffff, reduced, 1);
      reduced += __shfl_xor_sync(0xffffffff, reduced, 2);
      denominator[half] += reduced;
      maximum[half] = new_maximum[half];
    }
    if (tile != first_tile) {
#pragma unroll 1
      for (int chunk = 0; chunk < size<1>(thread_outputs); ++chunk) {
        auto fragment = make_tensor<float>(shape(thread_output_coordinates(_, _0{}, _0{}, _0{})));
        copy(output_copy, thread_outputs(_, chunk, _0{}, _0{}), fragment);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int element = 0; element < size(fragment); ++element) fragment(element) *= correction[element % 2];
        copy(output_store, fragment, thread_output_store(_, chunk, _0{}, _0{}));
      }
      cutlass::arch::fence_view_async_tmem_store();
    }
    cutlass::arch::cp_async_wait<0>();
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if (threadIdx.x < 32) {
      probability_value.accumulate_ = tile == first_tile ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
#pragma unroll
      for (int reduction = 0; reduction < size<2>(fragment_probability); ++reduction) {
        gemm(probability_value, fragment_probability(_, _, reduction), fragment_value(_, _, reduction), outputs);
        probability_value.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&storage.barrier);
    }
    cutlass::arch::ClusterBarrier::wait(&storage.barrier, barrier_phase);
    barrier_phase ^= 1;
    __syncthreads();
  }
  float* transposed_output = reinterpret_cast<float*>(storage.query.data());
  float inverse_denominator[2];
  float final_logsumexp[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    inverse_denominator[half] = denominator[half] > 0.0f ? 1.0f / denominator[half] : 0.0f;
    if constexpr (CATTN_SPLITS > 1) {
      final_logsumexp[half] = denominator[half] > 0.0f ? maximum[half] + log2f(denominator[half]) : -CUDART_INF_F;
    }
  }
#pragma unroll 1
  for (int chunk = 0; chunk < size<1>(thread_outputs); ++chunk) {
    auto fragment = make_tensor<float>(shape(thread_output_coordinates(_, _0{}, _0{}, _0{})));
    if (first_tile < last_tile) {
      copy(output_copy, thread_outputs(_, chunk, _0{}, _0{}), fragment);
      cutlass::arch::fence_view_async_tmem_load();
    } else {
      clear(fragment);
    }
    auto coordinates = thread_output_coordinates(_, chunk, _0{}, _0{});
#pragma unroll
    for (int element = 0; element < size(fragment); ++element) {
      auto coordinate = coordinates(element);
      int row = query_start + get<0>(coordinate);
      int column = get<1>(coordinate);
      transposed_output[int(get<0>(coordinate)) * 260 + column] = fragment(element) * inverse_denominator[element % 2];
      if constexpr (CATTN_SPLITS > 1) {
        if (row < query_rows * 8 && column == 0) logsumexp[blockIdx.y * query_rows * 8 + row] = final_logsumexp[element % 2];
      }
    }
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < 64 * 64; vector += 128) {
    int local_row = vector / 64;
    int column = (vector % 64) * 4;
    int row = query_start + local_row;
    const float* source = transposed_output + local_row * 260 + column;
    if (row < query_rows * 8) {
      float4 data = *reinterpret_cast<const float4*>(source);
      if constexpr (CATTN_SPLITS == 1) {
        cutlass::AlignedArray<Scalar, 4, 8> converted;
        converted[0] = Scalar(data.x);
        converted[1] = Scalar(data.y);
        converted[2] = Scalar(data.z);
        converted[3] = Scalar(data.w);
        *reinterpret_cast<uint2*>(direct_output + row * 256 + column) = *reinterpret_cast<const uint2*>(converted.data());
      } else {
        *reinterpret_cast<float4*>(partial + (blockIdx.y * query_rows * 8 + row) * 256 + column) = data;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) allocator.free(storage.tmem, 256);
}

__global__ void attention_cutlass_merge_v23(const float* partial, const float* logsumexp, Scalar* output, int rows) {
  int row = blockIdx.x * 8 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (row >= rows) return;
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int split = 0; split < CATTN_SPLITS; ++split) { maximum = fmaxf(maximum, logsumexp[split * rows + row]); }
  float numerator[8] = {};
  float denominator = 0.0f;
#pragma unroll
  for (int split = 0; split < CATTN_SPLITS; ++split) {
    float weight = maximum == -CUDART_INF_F ? 0.0f : sfu_exp2(logsumexp[split * rows + row] - maximum);
    denominator += weight;
#pragma unroll
    for (int vector = 0; vector < 2; ++vector) {
      float4 data = *reinterpret_cast<const float4*>(partial + (split * rows + row) * 256 + vector * 128 + lane * 4);
      numerator[vector * 4] += weight * data.x;
      numerator[vector * 4 + 1] += weight * data.y;
      numerator[vector * 4 + 2] += weight * data.z;
      numerator[vector * 4 + 3] += weight * data.w;
    }
  }
  float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
  for (int vector = 0; vector < 2; ++vector) {
    cutlass::AlignedArray<Scalar, 4, 8> converted;
#pragma unroll
    for (int element = 0; element < 4; ++element) converted[element] = Scalar(numerator[vector * 4 + element] * inverse);
    *reinterpret_cast<uint2*>(output + row * 256 + vector * 128 + lane * 4) = *reinterpret_cast<const uint2*>(converted.data());
  }
}

extern "C" int cattn01_prepare() {
  cudaError_t status =
      cudaFuncSetAttribute(attention_cutlass_tcgen_v20, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(AttentionStorage));
  if (status != cudaSuccess) return status;
  return cudaFuncSetAttribute(attention_cutlass_tcgen_v20, cudaFuncAttributePreferredSharedMemoryCarveout,
                              cudaSharedmemCarveoutMaxShared);
}

extern "C" int cattn01_launch(const void* query, const void* keys, const void* values, const int32_t* indptr,
                              const int32_t* indices, void* output, void* partial, void* logsumexp, int query_rows,
                              int64_t query_stride, int64_t head_stride, int64_t key_stride, int64_t value_stride, float scale,
                              int tile_rows, int tile_keys, cudaStream_t stream) {
  if (tile_rows != 64 || tile_keys != 128 || query_rows <= 0) return cudaErrorInvalidValue;
  attention_cutlass_tcgen_v20<<<dim3((query_rows * 8 + 63) / 64, CATTN_SPLITS), 128, sizeof(AttentionStorage), stream>>>(
      static_cast<const Scalar*>(query), static_cast<const Scalar*>(keys), static_cast<const Scalar*>(values), indptr, indices,
      static_cast<Scalar*>(output), static_cast<float*>(partial), static_cast<float*>(logsumexp), query_rows, query_stride,
      head_stride, key_stride, value_stride, scale);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if constexpr (CATTN_SPLITS == 1) return cudaSuccess;
  attention_cutlass_merge_v23<<<(query_rows * 8 + 7) / 8, 256, 0, stream>>>(
      static_cast<const float*>(partial), static_cast<const float*>(logsumexp), static_cast<Scalar*>(output), query_rows * 8);
  return cudaGetLastError();
}

#include "attention_ffi.cuh"
