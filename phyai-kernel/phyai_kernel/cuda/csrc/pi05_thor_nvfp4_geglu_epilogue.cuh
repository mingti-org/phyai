// SPDX-License-Identifier: Apache-2.0
// Compact GeGLU and NVFP4 output callback for the fixed pi0.5 Thor kernel.

#pragma once

#include <cstdint>

#include <cuda_fp8.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp>

namespace phyai::pi05::thor::detail {

using namespace cute;

template <int kVectorSize, class EpilogueTile, class ElementOutput, class ElementCompute, class ElementScale,
          cutlass::FloatRoundStyle kRoundStyle = cutlass::FloatRoundStyle::round_to_nearest>
struct CompactGeGLUStoreNode {
  static_assert(size<1>(EpilogueTile{}) % (2 * kVectorSize) == 0,
                "the epilogue tile must contain complete gate/up scale blocks");

  struct SharedStorage {};

  struct Arguments {
    uint8_t* output = nullptr;
    uint8_t* output_scale = nullptr;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& arguments, void*) {
    return arguments;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& arguments) {
    auto shape = append<4>(problem_shape, 1);
    const auto columns = get<1>(shape);
    return columns % (2 * kVectorSize) == 0 && arguments.output != nullptr && arguments.output_scale != nullptr;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE CompactGeGLUStoreNode() = default;

  CUTLASS_HOST_DEVICE CompactGeGLUStoreNode(Params const& params, SharedStorage const&) : params_(&params) {}

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }

  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(cutlass::epilogue::fusion::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  template <class CoordinateTensor, class Residue, class ScaleLayout>
  struct StoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE StoreCallbacks(CoordinateTensor coordinates, Residue residue, ScaleLayout scale_layout,
                                  Params const* params, int row_bytes, int tile_row, int tile_column)
        : coordinates_(coordinates),
          residue_(residue),
          scale_layout_(scale_layout),
          params_(params),
          row_bytes_(row_bytes),
          tile_row_(tile_row),
          tile_column_(tile_column) {}

    template <class ElementAccumulator, class ElementInput, int kFragmentSize>
    CUTLASS_DEVICE auto visit(cutlass::Array<ElementAccumulator, kFragmentSize> const&, int, int epi_m, int epi_n,
                              cutlass::Array<ElementInput, kFragmentSize> const& input) {
      static_assert(kFragmentSize % (2 * kVectorSize) == 0,
                    "each callback fragment must contain complete gate/up scale blocks");
      constexpr int kBlocks = kFragmentSize / (2 * kVectorSize);

      auto fragment_coordinates = coordinates_(_, _, _, epi_m, epi_n);
      const auto first = fragment_coordinates(0);
      if (elem_less(first, residue_)) {
        const int row = tile_row_ + get<0>(first);
        const int merged_column = tile_column_ + get<1>(first);
        uint8_t* packed = params_->output + static_cast<int64_t>(row) * row_bytes_ + merged_column / 4;

        CUTLASS_PRAGMA_UNROLL
        for (int block = 0; block < kBlocks; ++block) {
          const int compact_column = merged_column / 2 + block * kVectorSize;
          cutlass::Array<ElementCompute, kVectorSize> values;
          float maximum = 0.0f;

          CUTLASS_PRAGMA_UNROLL
          for (int index = 0; index < kVectorSize; ++index) {
            const float gate = static_cast<float>(input[block * 2 * kVectorSize + 2 * index]);
            const float up = static_cast<float>(input[block * 2 * kVectorSize + 2 * index + 1]);
            constexpr float kCubicCoefficient = 0.044715f;
            constexpr float kTanhCoefficient = 1.5957691216057308f;
            const float activated =
                gate / (1.0f + expf(-kTanhCoefficient * gate * (1.0f + kCubicCoefficient * gate * gate)));
            const float value = activated * up;
            values[index] = static_cast<ElementCompute>(value);
            maximum = maximum > fabsf(value) ? maximum : fabsf(value);
          }

          float output_scale = maximum / 6.0f;
          output_scale = output_scale > 1.0e-12f ? output_scale : 1.0e-12f;
          __nv_fp8_e4m3 quantized_scale(output_scale);
          params_->output_scale[scale_layout_(row, compact_column, 0)] = quantized_scale.__x;
          const float inverse_scale = 1.0f / static_cast<float>(quantized_scale);

          CUTLASS_PRAGMA_UNROLL
          for (int index = 0; index < kVectorSize; ++index) {
            values[index] = static_cast<ElementCompute>(values[index] * inverse_scale);
          }

          auto packed_values =
              cutlass::NumericArrayConverter<cutlass::float_e2m1_t, ElementCompute, kVectorSize, kRoundStyle>{}(values);
          *reinterpret_cast<uint2*>(packed + block * (kVectorSize / 2)) =
              *reinterpret_cast<uint2 const*>(&packed_values);
        }
      }

      cutlass::Array<ElementOutput, kFragmentSize> ignored_output;
      ignored_output.fill(ElementOutput(0));
      return ignored_output;
    }

    CoordinateTensor coordinates_;
    Residue residue_;
    ScaleLayout scale_layout_;
    Params const* params_;
    int row_bytes_;
    int tile_row_;
    int tile_column_;
  };

  template <bool, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    auto [m, n, k, l] = args.problem_shape_mnkl;
    using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kVectorSize>;
    auto scale_layout = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, 1, n / 2, 1));
    return StoreCallbacks(args.tCcD, args.residue_tCcD, scale_layout, params_, n / 4,
                          m - static_cast<int>(get<0>(args.residue_tCcD)),
                          n - static_cast<int>(get<1>(args.residue_tCcD)));
  }

 private:
  Params const* params_ = nullptr;
};

template <int kVectorSize, class ElementOutput, class ElementCompute, class ElementScale, class ScaleLayout,
          class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
          cutlass::FloatRoundStyle kRoundStyle = cutlass::FloatRoundStyle::round_to_nearest>
struct CompactGeGLUFusion
    : cutlass::epilogue::fusion::LinCombBlockScaleFactor<kVectorSize, ElementOutput, ElementCompute, ElementScale,
                                                         ScaleLayout, ElementSource, ElementScalar, kRoundStyle> {};

}  // namespace phyai::pi05::thor::detail

namespace cutlass::epilogue::fusion {

template <int kVectorSize, class EpilogueTile, class ElementOutput, class ElementCompute, class ElementScale,
          class ElementSource, class ElementScalar, cutlass::FloatRoundStyle kRoundStyle>
using Pi05CompactGeGLUTree =
    Sm90EVT<phyai::pi05::thor::detail::CompactGeGLUStoreNode<kVectorSize, EpilogueTile, ElementOutput, ElementCompute,
                                                            ElementScale, kRoundStyle>,
            Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, kRoundStyle>>;

template <int kStagesC, int kStagesD, int kFragmentSize, bool kReuseSmemC, bool kDelayTmaStore, class ElementOutput,
          class ElementCompute, class ElementScale, int kVectorSize, class ElementSource, class ElementScalar,
          cutlass::FloatRoundStyle kRoundStyle, class CtaTile, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<kStagesC, kStagesD, kFragmentSize, kReuseSmemC, kDelayTmaStore>,
    phyai::pi05::thor::detail::CompactGeGLUFusion<kVectorSize, ElementOutput, ElementCompute, ElementScale,
                                                 cutlass::layout::RowMajor, ElementSource, ElementScalar, kRoundStyle>,
    CtaTile, EpilogueTile>
    : Pi05CompactGeGLUTree<kVectorSize, EpilogueTile,
                           typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute,
                           ElementScale, ElementSource, ElementScalar, kRoundStyle> {
  using Impl = Pi05CompactGeGLUTree<kVectorSize, EpilogueTile,
                                    typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
                                    ElementCompute, ElementScale, ElementSource, ElementScalar, kRoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    using ScalarStride = cute::Stride<cute::_0, cute::_0, int64_t>;
    ScalarStride alpha_stride = {cute::_0{}, cute::_0{}, 0};
    ScalarStride beta_stride = {cute::_0{}, cute::_0{}, 0};
    uint8_t* output = nullptr;
    uint8_t* output_scale = nullptr;

    operator typename Impl::Arguments() const {
      return {{{{beta}, {beta_ptr}, {beta_stride}},
               {},
               {{{alpha}, {alpha_ptr}, {alpha_stride}}, {}, {}},
               {}},
              {output, output_scale}};
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion
