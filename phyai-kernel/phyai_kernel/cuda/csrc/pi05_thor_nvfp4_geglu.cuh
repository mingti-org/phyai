// SPDX-License-Identifier: Apache-2.0
// Fixed-shape SM110 NVFP4 Gate+Up GEMM with compact GeGLU output for pi0.5.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#include <tvm/ffi/container/tensor.h>

#include "pi05_thor_nvfp4_geglu_epilogue.cuh"

namespace phyai::pi05::thor {

using namespace cute;

using MmaTile = Shape<_128, _256, _256>;
using Cluster = Shape<_1, _2, _1>;
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::float_e2m1_t;
using ElementC = ElementD;
using ElementSFD = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentC = 32;
constexpr int kAlignmentD = 32;
constexpr int kScaleVector = 16;

using Fusion = detail::CompactGeGLUFusion<kScaleVector, ElementD, ElementCompute, ElementSFD,
                                          cutlass::layout::RowMajor, ElementC>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
    ElementAccumulator, ElementC, cutlass::layout::RowMajor, kAlignmentC, ElementD, cutlass::layout::RowMajor, kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, cutlass::layout::RowMajor, kAlignmentA, ElementB, cutlass::layout::ColumnMajor,
    kAlignmentB, ElementAccumulator, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Kernel::StrideA;
using StrideB = typename Kernel::StrideB;
using StrideC = typename Kernel::StrideC;
using StrideD = typename Kernel::StrideD;
using BlockScaledConfig = typename Mainloop::Sm1xxBlkScaledConfig;

inline int run(const void* activation, const void* activation_scale, const void* interleaved_weight, const void* weight_scale,
               void* workspace, void* output, void* output_scale, int m, int n, int k, cudaStream_t stream) {
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_sfa = BlockScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb = BlockScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  using AData = typename ElementA::DataType;
  using AScale = typename ElementA::ScaleFactorType;
  using BData = typename ElementB::DataType;
  using BScale = typename ElementB::ScaleFactorType;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<const AData*>(activation), stride_a, reinterpret_cast<const BData*>(interleaved_weight), stride_b,
       reinterpret_cast<const AScale*>(activation_scale), layout_sfa, reinterpret_cast<const BScale*>(weight_scale),
       layout_sfb},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC*>(workspace), stride_c, reinterpret_cast<ElementD*>(workspace), stride_d}};
  arguments.epilogue.thread.output = reinterpret_cast<uint8_t*>(output);
  arguments.epilogue.thread.output_scale = reinterpret_cast<uint8_t*>(output_scale);

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) { return static_cast<int>(status) | 0x10000; }
  if (Gemm::get_workspace_size(arguments) != 0) { return -3; }
  status = gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) { return static_cast<int>(status) | 0x20000; }
  status = gemm.run(stream);
  return status == cutlass::Status::kSuccess ? 0 : (static_cast<int>(status) | 0x30000);
}

}  // namespace phyai::pi05::thor

extern "C" int phyai_pi05_thor_nvfp4_geglu(tvm::ffi::TensorView activation, tvm::ffi::TensorView activation_scale,
                                           tvm::ffi::TensorView interleaved_weight, tvm::ffi::TensorView weight_scale,
                                           tvm::ffi::TensorView workspace, tvm::ffi::TensorView output,
                                           tvm::ffi::TensorView output_scale, int64_t m, int64_t stream_handle) {
  if ((m != 784 && m != 816 && m != 880 && m != 968) || activation.size(0) != m) { return -2; }
  auto data = [](const tvm::ffi::TensorView& tensor) {
    return static_cast<void*>(static_cast<char*>(tensor.data_ptr()) + tensor.byte_offset());
  };
  return phyai::pi05::thor::run(data(activation), data(activation_scale), data(interleaved_weight), data(weight_scale),
                                data(workspace), data(output), data(output_scale), static_cast<int>(m), 32768, 2048,
                                reinterpret_cast<cudaStream_t>(stream_handle));
}
