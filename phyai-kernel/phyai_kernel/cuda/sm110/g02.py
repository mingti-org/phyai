# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Type, Tuple, Union
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class GemmBlackwellFP16:
    """Blackwell SM100 dense GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

    FP16 inputs/outputs with FP32 accumulation via tcgen05.mma (UMMA).
    TMA store epilogue (cp.bulk.tensor.s2g) with auto box-bounds.

    Configurable per construction:
      mma_tiler_mn:     per-CTA tile (default (64, 128); use (128, 128) for
                        large M >= 1024)
      cluster_shape_mn: cluster size for TMA multicast (default (1, 2)
                        multicasts B across two CTAs in N)
      use_2cta:         pair two CTAs along M for a 2x M tile per UMMA op
                        (requires cluster[0] >= 2; not used by current AOT)
      mma_inst_tile_k:  K-tile multiplier per MMA instruction (default 4)

    Designed for the Qwen3-Omni Talker MLP where M is the token count
    (dynamic) and N, K are weight dimensions (also dynamic for AOT).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mn: Tuple[int, int] = (64, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 2),
        use_2cta: bool = False,
        mma_inst_tile_k: int = 4,
        ab_stages: int = 8,
    ):
        self.acc_dtype = acc_dtype
        # 2-CTA MMA pairs two CTAs along the M dim and issues a single
        # tcgen05.mma instruction that covers a 2x larger M tile per cta_group.
        # Requires cluster_shape[0] >= 2 so the leader CTA can drive the peer.
        self.use_2cta_instrs = use_2cta
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        # K-tile multiplier per MMA instruction K-shape. Larger -> more K
        # accumulated per MMA op (longer K iter span, fewer pipeline syncs).
        self.mma_inst_tile_k = mma_inst_tile_k
        # TMA store: TMEM -> regs -> SMEM -> gmem via cp.bulk.tensor.s2g.
        # The TMA descriptor handles partial-tile boundaries automatically,
        # so no manual M/N predication is needed in the epilogue.
        self.use_tma_store = False

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta else tcgen05.CtaGroup.ONE
        self.ab_stages_override = ab_stages
        self.occupancy = 1
        self.threads_per_cta = 192

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = (self.cta_tile_shape_mnk[0], 8)
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]

        # cuTe DSL 4.5.2 does not reliably auto-detect Thor's Blackwell family
        # SMs (e.g. SM101/SM110) in get_smem_capacity_in_bytes(). Like the FMHA
        # Blackwell path, request the Blackwell-family capacity explicitly.
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
        )

        self.num_ab_stage = self.ab_stages_override
        self.num_c_stage = 1

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        use_silu: cutlass.Constexpr = False,
        mBias: cute.Tensor = None,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype mismatch: {self.a_dtype} vs {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.make_composed_layout(
            self.b_smem_layout_staged.inner,
            0,
            cute.slice_(self.b_smem_layout_staged.outer, (None, None, None, 0)),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # TMA store atom for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            mBias,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            use_silu,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=True,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        mBias: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        use_silu: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1],
            cta_coord[2],
        )
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 128
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )

        # Barrier id 0 is reserved by other driver APIs (e.g. sync_threads()).
        # Use a non-zero named barrier to avoid runtime warnings and potential
        # synchronization conflicts.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.threads_per_cta
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # SMEM tensors. sC is only allocated for TMA store (R2S staging).
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Identity tensor mirroring tCgC so the epilogue can derive per-element
        # absolute (M, N) coordinates that physically align with tCgC.
        mcC_id = cute.make_identity_tensor(mC_mnl.layout.shape)
        gC_id_mnl = cute.local_tile(
            mcC_id, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(gC_id_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # Allocation runs on warp1 while warp0 primes the empty input ring.
        # No stage is reused before the consumer starts; all peers prime the
        # same first stage after the original cluster-init barrier.
        tmem.allocate(self.num_tmem_alloc_cols)
        prologue_count = 1
        if warp_idx == 4:
            cute.arch.griddepcontrol_wait()
            for k_tile_idx in cutlass.range(prologue_count, unroll_full=True):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_idx)],
                    tAsA[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_idx)],
                    tBsB[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # Independent producer and consumer warps keep the input ring full.
        # Pipeline barriers, expected transaction bytes, phase advancement and
        # consumer releases retain the official PipelineTmaUmma contract.
        if warp_idx == 4:
            for k_tile_idx in cutlass.range(
                prologue_count, k_tile_cnt, unroll_full=True
            ):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_idx)],
                    tAsA[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_idx)],
                    tBsB[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
            # Drain the input ring in the producer warp while other warps wait
            # for the accumulator. Keep all releases and cross-CTA waits.
            ab_producer.tail()

        if warp_idx == 5:
            cute.arch.griddepcontrol_launch_dependents()
            if is_leader_cta:
                for k_tile_idx in cutlass.range(k_tile_cnt, unroll_full=True):
                    consumer_handle = ab_consumer.wait_and_advance()
                    num_kblks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    consumer_handle.release()
                acc_pipeline.producer_commit(acc_producer_state)

        # ---- Epilogue: TMEM -> registers (-> SMEM -> gmem via TMA) ----
        tmem.relinquish_alloc_permit()
        if warp_idx < 4:
            acc_pipeline.consumer_wait(acc_consumer_state)

            if cutlass.const_expr(self.use_tma_store):
                self.epilogue_tma_store(
                    tidx,
                    warp_idx,
                    mma_tile_coord_mnl,
                    tma_atom_c,
                    tCtAcc,
                    sC,
                    tCgC,
                    tCcC,
                    mC_mnl,
                    epi_tile,
                    mBias,
                    use_silu,
                )
            else:
                epi_tidx = tidx
                assert mBias is None and not use_silu
                assert not self.use_2cta_instrs
                # Full-CTA native TMEM ownership remains unchanged. Stage the BF16
                # result, then repartition it independently for vector global stores.
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    self.c_layout,
                    self.c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    self.use_2cta_instrs,
                )
                tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
                )
                thr_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tCgC_epi = cute.flat_divide(
                    tCgC[((None, None), 0, 0, None, None, None)], epi_tile
                )
                tTR_gC = thr_t2r.partition_D(tCgC_epi)
                rAcc = cute.make_rmem_tensor(
                    tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
                )
                rC = cute.make_rmem_tensor(rAcc.shape, self.c_dtype)
                atom_r2s = sm100_utils.get_smem_store_op(
                    self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
                )
                tiled_r2s = cute.make_tiled_copy_D(atom_r2s, tiled_copy_t2r)
                thr_r2s = tiled_r2s.get_slice(epi_tidx)
                tRS_sC = thr_r2s.partition_D(sC)
                tRS_rC = tiled_r2s.retile(rC)
                assert cute.size(tTR_tAcc.shape, mode=[3]) == 1
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0)], rAcc)
                tRS_rC.store(tiled_r2s.retile(rAcc).load().to(self.c_dtype))
                cute.copy(tiled_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
                pipeline.NamedBarrier(barrier_id=1, num_threads=128).sync()
                # Every TMEM reader completed its read and BF16 staging.
                # All subsequent output accesses use the shared buffer only.
                tmem.free(tmem_ptr)
                # M is contiguous in the transposed logical output. Every vector is
                # eight BF16 values; all three fixed shapes have complete M tiles.
                atom_s2g = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
                )
                threads_m = self.cta_tile_shape_mnk[0] // 8
                thr_layout = cute.make_layout(
                    (threads_m, 128 // threads_m), stride=(1, threads_m)
                )
                val_layout = cute.make_layout((8, 1), stride=(1, 8))
                tiled_s2g = cute.make_tiled_copy_tv(atom_s2g, thr_layout, val_layout)
                thr_s2g = tiled_s2g.get_slice(epi_tidx)
                sC0 = sC[(None, None, 0)]
                gC = cute.local_tile(
                    mC_mnl[(None, None, mma_tile_coord_mnl[2])],
                    self.cta_tile_shape_mnk[:2],
                    mma_tile_coord_mnl[:2],
                )
                coords = cute.local_tile(
                    cute.make_identity_tensor(mC_mnl.shape[:2]),
                    self.cta_tile_shape_mnk[:2],
                    mma_tile_coord_mnl[:2],
                )
                tSG_sC = thr_s2g.partition_S(sC0)
                tSG_gC = thr_s2g.partition_D(gC)
                tSG_cC = thr_s2g.partition_D(coords)
                for rm in cutlass.range(cute.size(tSG_gC, mode=[1]), unroll_full=True):
                    for rn in cutlass.range(
                        cute.size(tSG_gC, mode=[2]), unroll_full=True
                    ):
                        coord = tSG_cC[(0, rm, rn)]
                        if (
                            coord[0] + 7 < mC_mnl.shape[0]
                            and coord[1] < mC_mnl.shape[1]
                        ):
                            cute.copy(
                                atom_s2g, tSG_sC[(None, rm, rn)], tSG_gC[(None, rm, rn)]
                            )

        # All accumulator readers completed their epilogue. The output
        # allocator warp frees TMEM after a128-reader barrier; the complete
        # CTA then drains input producers before exit.
        pipeline.sync(barrier_id=2)

        return

    @cute.jit
    def epilogue_tma_store(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tma_atom_c: cute.CopyAtom,
        tCtAcc: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        tCcC: cute.Tensor,
        mC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mBias: cute.Tensor,
        use_silu: cutlass.Constexpr,
    ) -> None:
        """TMA-store epilogue: TMEM -> regs -> SMEM -> gmem via cp.bulk.tensor.

        Adapted from cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py
        with the bias-add and SiLU activation hooks added in the register
        stage (between T2R and R2S).
        """
        # T2R: TMEM -> registers (per-thread fragment).
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

        # tCgC partition for absolute coords (used for bias N indexing).
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        # Mirror tCgC's partition path on the identity coord tensor so we can
        # gate bias loads by the absolute N coord.
        tCcC_epi = cute.flat_divide(
            tCcC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_cC = thr_copy_t2r.partition_D(tCcC_epi)
        tTR_cC = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

        actual_N = mC_mnl.layout.shape[1]

        # Bias broadcast tile (gmem). Same construction as the simt path.
        if cutlass.const_expr(mBias is not None):
            n_tile_offset = mma_tile_coord_mnl[1] * cutlass.Int32(self.mma_tiler[1])
            gBias_tile = cute.make_tensor(
                mBias.iterator + n_tile_offset,
                cute.make_layout((self.mma_tiler[0], self.mma_tiler[1]), stride=(0, 1)),
            )
            tCgBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tTR_gBias = thr_copy_t2r.partition_D(tCgBias_epi)
            tTR_gBias = cute.group_modes(tTR_gBias, 3, cute.rank(tTR_gBias))

        # R2S: registers -> SMEM (staging buffer for TMA store).
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(epi_tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)

        # TMA store partition: SMEM -> gmem. The TMA descriptor itself
        # handles partial-tile boundaries — no manual M/N predication needed.
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage, producer_group=c_producer_group
        )

        output_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            coord_slice = tTR_cC[(None, None, None, subtile_idx)]

            # Bias add (broadcast over M, indexed by N), gated by actual_N.
            if cutlass.const_expr(mBias is not None):
                bias_slice = tTR_gBias[(None, None, None, subtile_idx)]
                for i in cutlass.range(cute.size(tTR_rAcc)):
                    if coord_slice[i][1] < actual_N:
                        tTR_rAcc[i] = tTR_rAcc[i] + bias_slice[i].to(self.acc_dtype)
            if cutlass.const_expr(use_silu):
                for si in cutlass.range(cute.size(tTR_rAcc)):
                    val = tTR_rAcc[si]
                    tTR_rAcc[si] = val * cute.arch.rcp_approx(
                        1.0 + cute.exp(-val, fastmath=True)
                    )

            # Convert acc -> c_dtype in registers.
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(self.c_dtype))

            # R2S into the c-stage SMEM buffer.
            c_buffer = subtile_idx % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            output_barrier.sync()

            # TMA S2G — only warp 0 issues the store.
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)]
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            output_barrier.sync()

        c_pipeline.producer_tail()

    @cute.jit
    def epilogue(
        self,
        epi_tidx: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tCtAcc: cute.Tensor,
        sC: cute.Tensor,
        tCgC: cute.Tensor,
        tCcC: cute.Tensor,
        mC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        mBias: cute.Tensor,
        use_silu: cutlass.Constexpr,
    ) -> None:
        assert mBias is None and not use_silu
        assert not self.use_2cta_instrs
        # Full-CTA native TMEM ownership remains unchanged. Stage the BF16
        # result, then repartition it independently for vector global stores.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_t2r.partition_D(tCgC_epi)
        rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        rC = cute.make_rmem_tensor(rAcc.shape, self.c_dtype)
        atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_r2s = cute.make_tiled_copy_D(atom_r2s, tiled_copy_t2r)
        thr_r2s = tiled_r2s.get_slice(epi_tidx)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rC = tiled_r2s.retile(rC)
        assert cute.size(tTR_tAcc.shape, mode=[3]) == 1
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0)], rAcc)
        tRS_rC.store(tiled_r2s.retile(rAcc).load().to(self.c_dtype))
        cute.copy(tiled_r2s, tRS_rC, tRS_sC[(None, None, None, 0)])
        pipeline.NamedBarrier(barrier_id=1, num_threads=128).sync()
        # M is contiguous in the transposed logical output. Every vector is
        # eight BF16 values; all three fixed shapes have complete M tiles.
        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.c_dtype, num_bits_per_copy=128
        )
        threads_m = self.cta_tile_shape_mnk[0] // 8
        thr_layout = cute.make_layout(
            (threads_m, 128 // threads_m), stride=(1, threads_m)
        )
        val_layout = cute.make_layout((8, 1), stride=(1, 8))
        tiled_s2g = cute.make_tiled_copy_tv(atom_s2g, thr_layout, val_layout)
        thr_s2g = tiled_s2g.get_slice(epi_tidx)
        sC0 = sC[(None, None, 0)]
        gC = cute.local_tile(
            mC_mnl[(None, None, mma_tile_coord_mnl[2])],
            self.cta_tile_shape_mnk[:2],
            mma_tile_coord_mnl[:2],
        )
        coords = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape[:2]),
            self.cta_tile_shape_mnk[:2],
            mma_tile_coord_mnl[:2],
        )
        tSG_sC = thr_s2g.partition_S(sC0)
        tSG_gC = thr_s2g.partition_D(gC)
        tSG_cC = thr_s2g.partition_D(coords)
        for rm in cutlass.range(cute.size(tSG_gC, mode=[1]), unroll_full=True):
            for rn in cutlass.range(cute.size(tSG_gC, mode=[2]), unroll_full=True):
                coord = tSG_cC[(0, rm, rn)]
                if coord[0] + 7 < mC_mnl.shape[0] and coord[1] < mC_mnl.shape[1]:
                    cute.copy(atom_s2g, tSG_sC[(None, rm, rn)], tSG_gC[(None, rm, rn)])

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        c_layout,
        smem_capacity,
        occupancy,
        use_tma_store,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1
        num_c_stage = 2 if use_tma_store else 0

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        c_smem_layout_staged_one = (
            sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
            if use_tma_store
            else None
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = (
            cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
            if use_tma_store
            else 0
        )
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        if use_tma_store:
            num_c_stage += (
                smem_capacity
                - ab_bytes_per_stage * num_ab_stage
                - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)
            ) // ((occupancy + 1) * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(c, cta_tile_shape_mnk, cluster_shape_mn):
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]),
                cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]),
                c.layout.shape[2],
            ),
            cluster_shape_mnl,
        )
        return grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)
