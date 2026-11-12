/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "compact_attention/fmha_cutlass_sm100.cuh"
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;


using namespace flashinfer;
void run(ffi::TensorView workspace_buffer, ffi::TensorView q, ffi::TensorView k,
                         ffi::TensorView v, ffi::TensorView qo_segment_offsets,
                         ffi::TensorView kv_segment_offsets, ffi::TensorView work_indptr,
                         ffi::TensorView qo_tile_indices, ffi::TensorView qo_head_indices,
                         ffi::TensorView batch_indices, ffi::TensorView o,
                         Optional<ffi::TensorView> maybe_lse, int64_t mask_mode_code,
                         double sm_scale, double scale_q, double scale_k, double scale_v,
                         double o_scale, int64_t max_qo_len) {
  TVM_FFI_ICHECK_EQ(q.dtype(), k.dtype());
  TVM_FFI_ICHECK_EQ(q.dtype(),(DLDataType{kDLBfloat,16,1}));
  TVM_FFI_ICHECK(q.size(2)==128 && v.size(2)==192 && mask_mode_code==0);
  auto scalar_type_in = q.dtype();
  auto scalar_type_out = o.dtype();
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  int total_qo_len = q.size(0);
  int total_kv_len = k.size(0);
  int num_qo_heads = q.size(1);
  int num_kv_heads = k.size(1);
  int head_dim_qk = q.size(2);
  int head_dim_vo = v.size(2);
  int batch_size = qo_segment_offsets.size(0) - 1;
  int q_stride_n = q.stride(0);
  int q_stride_h = q.stride(1);
  int k_stride_n = k.stride(0);
  int k_stride_h = k.stride(1);
  int v_stride_n = v.stride(0);
  int v_stride_h = v.stride(1);

  ffi::CUDADeviceGuard device_guard(qo_segment_offsets.device().device_id);
  const cudaStream_t stream = get_stream(o.device());

    using cutlass_type_in = cutlass::bfloat16_t;
    using cutlass_type_out = cutlass::bfloat16_t;
    using TILE_Q = _256;
    using TILE_KV = _64;
    using D_QK = cute::Int<128>;
    using D_VO = cute::Int<192>;
    using TileShapeQK = Shape<TILE_Q, TILE_KV, D_QK>;
    using TileShapePV = Shape<TILE_Q, D_VO, TILE_KV>;
    using CutlassMaskMode = ResidualMask;
    auto status = run_fmha_fwd<cutlass_type_in, cutlass_type_out, int32_t, TileShapeQK, TileShapePV,
                               CutlassMaskMode>(
        workspace_buffer.data_ptr(), static_cast<cutlass_type_in*>(q.data_ptr()),
        static_cast<cutlass_type_in*>(k.data_ptr()), static_cast<cutlass_type_in*>(v.data_ptr()),
        static_cast<int*>(qo_segment_offsets.data_ptr()),
        static_cast<int*>(kv_segment_offsets.data_ptr()), static_cast<int*>(work_indptr.data_ptr()),
        static_cast<int*>(qo_tile_indices.data_ptr()),
        static_cast<int*>(qo_head_indices.data_ptr()), static_cast<int*>(batch_indices.data_ptr()),
        static_cast<cutlass_type_out*>(o.data_ptr()),
        maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr,
        mask_mode_code, sm_scale, scale_q, scale_k, scale_v, o_scale, num_qo_heads, num_kv_heads,
        head_dim_qk, head_dim_vo, q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n,
        v_stride_h, batch_size, total_qo_len, total_kv_len, max_qo_len, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "Cutlass FMHA forward pass failed" << cudaGetErrorString(status);

}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run,run);
