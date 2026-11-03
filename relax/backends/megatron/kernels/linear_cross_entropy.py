# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Adapted from the Megatron Core linear_cross_entropy/blackwell kernels used by Miles.
# Original Megatron source license:
# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Miles SM100 linear CE for Relax: per-token loss, already-gathered SP
rows."""

import os
import typing
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Tuple, Type

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from relax.utils.logging_utils import get_logger


# TODO(remove): Megatron dev already provides linear CE. Remove this temporary
# Miles kernel backport when the pinned MCore dependency is upgraded and its
# native forward/backward path is validated.


_KERNEL_IMPORT_ERROR: ImportError | None = None

# This module is loaded only when RELAX_SFT_FUSED_LINEAR_CE is enabled. Stream
# creation and compilation remain deferred until the first kernel invocation.
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as cutlass_utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import triton
    import triton.language as tl
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack

    # Preserve the API alias used by the historical Miles CE runtime.
    if not hasattr(cute, "make_fragment"):
        cute.make_fragment = cute.make_rmem_tensor

    SM100_TMEM_CAPACITY_COLUMNS: int = 512

    # Triton reduction kernels.

    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
        key=["num_tokens", "num_splits"],
    )
    @triton.jit
    def forward_dp_epilogue(
        num_tokens: tl.int64,
        num_splits: tl.int64,  # TODO: maybe this could be a constexpr
        ignore_index: tl.int64,
        labels_ptr,  #: tl.pointer_type(tl.int64),
        stride_labels: tl.int64,
        max_ptr,  #: tl.pointer_type(tl.float32),
        stride_max_m: tl.int64,
        stride_max_n: tl.int64,
        accu_ptr,  #: tl.pointer_type(tl.float32),
        stride_accu_m: tl.int64,
        stride_accu_n: tl.int64,
        global_max_ptr,  #: tl.pointer_type(tl.float32),
        stride_global_max: tl.int64,
        global_accu_ptr,  #: tl.pointer_type(tl.float32),
        stride_global_accu: tl.int64,
        global_logprobs_ptr,  #: tl.pointer_type(tl.float32),
        stride_global_logprobs: tl.int64,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            _max = tl.load(
                max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )
            _accu = tl.load(
                accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )

            # local reduction
            _max_old = global_max
            _local_max = tl.max(_max, axis=1, return_indices=False)
            global_max = tl.maximum(global_max, _local_max)

            _scale = tl.exp(_max - global_max[:, None])
            _coeff = tl.exp(_max_old - global_max)
            global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

        # store maximum
        tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
        # store accumulate
        tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)
        # update logprobs
        labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index)
        global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
        global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
        global_logprobs = global_max + tl.log(global_accu) - global_logprobs
        label_mask = labels != ignore_index
        global_logprobs = tl.where(label_mask, global_logprobs, 0.0)

        tl.store(global_logprobs_ptrs, global_logprobs, mask=offs_m < num_tokens)

    @triton.autotune(
        configs=[triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})],
        key=["num_tokens", "num_splits"],
    )
    @triton.jit
    def forward_tp_epilogue(
        num_tokens: tl.int64,
        num_splits: tl.int64,
        reduced_max_ptr,  #: tl.pointer_type(tl.float32),
        stride_reduced_max_m: tl.int64,
        stride_reduced_max_n: tl.int64,
        original_max_ptr,  #: tl.pointer_type(tl.float32),
        stride_original_max_m: tl.int64,
        stride_original_max_n: tl.int64,
        accu_ptr,  #: tl.pointer_type(tl.float32),
        stride_accu_m: tl.int64,
        stride_accu_n: tl.int64,
        global_max_ptr,  #: tl.pointer_type(tl.float32),
        stride_global_max: tl.int64,
        global_accu_ptr,  #: tl.pointer_type(tl.float32),
        stride_global_accu: tl.int64,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        global_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            _reduced_max = tl.load(
                reduced_max_ptr + offs_m[:, None] * stride_reduced_max_m + offs_n[None, :] * stride_reduced_max_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )
            _original_max = tl.load(
                original_max_ptr + offs_m[:, None] * stride_original_max_m + offs_n[None, :] * stride_original_max_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )
            _accu = tl.load(
                accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )

            # local reduction
            _max_old = global_max
            _local_max = tl.max(_reduced_max, axis=1)
            global_max = tl.maximum(global_max, _local_max)

            # update accumulate
            _coeff = tl.exp(_max_old - global_max)
            _scale = tl.exp(_original_max - global_max[:, None])
            global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

        # store
        tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
        tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 16})], key=["num_tokens"])
    @triton.jit
    def forward_tp_epilogue_update_logprobs(
        num_tokens: tl.int64,
        ignore_index: tl.int64,
        labels_ptr,  #: tl.pointer_type(tl.int64),
        stride_labels: tl.int64,
        logprobs_ptr,  #: tl.pointer_type(tl.float32),
        stride_logprobs: tl.int64,
        maximum_ptr,  #: tl.pointer_type(tl.float32),
        stride_maximum: tl.int64,
        accumulate_ptr,  #: tl.pointer_type(tl.float32),
        stride_accumulate: tl.int64,
        BLOCK_SIZE_M: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
        maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
        accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)

        labels = tl.load(labels_ptr + offs_m * stride_labels, mask=offs_m < num_tokens, other=ignore_index)
        label_mask = labels != ignore_index

        logprobs = maximum + tl.log(accumulate) - logprobs
        logprobs = tl.where(label_mask, logprobs, 0.0)

        tl.store(logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens)

    # CuTe forward mainloop.
    def _thread_group(size: int):
        return pipeline.CooperativeGroup(pipeline.Agent.Thread, size, alignment=size)

    class FwdMainLoop:
        def __init__(self, vocab_per_split: int):
            self.acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
            self.use_2cta_instrs = False
            # This is the shape covered by tiledMMA, not just single MMA instruction
            self.mma_tiler = (128, 256, 1)
            self.cta_tiler = (self.mma_tiler[0], vocab_per_split, self.mma_tiler[2])
            self.vocab_per_split = vocab_per_split

            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)

            # the maximum columns per MMA is 256, and there is only one GEMM, so we can fully
            # assign TMEM for that GEMM of different tiles.
            # so 512 = 2 * 256

            self.threads_per_warp: int = 32
            # 1 warp for loading, 1 warp for issuing MMA, 1 WG for storing
            self.epi_warp_ids = (0, 1, 2, 3)
            self.load_warp_ids = 4
            self.mma_warp_ids = 5
            self.empty_warp_ids = (6, 7)

            self.threads_per_cta: int = self.threads_per_warp * len(
                (*self.epi_warp_ids, self.load_warp_ids, self.mma_warp_ids, *self.empty_warp_ids)
            )

            self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)

            self.buffer_align_bytes: int = 1024
            self.num_regs_other: int = 32
            self.num_regs_epi: int = 192

        def _setup_attributes(self, tiled_mma: cute.TiledMma):
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,)
            )

            # this is fixed for dense MMA, k=16
            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            # 16*4 = 64; 64 * sizeof(FP16) = 128Bytes
            mma_inst_tile_k: int = 4
            self.mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )

            self.num_acc_stage, self.num_a_stage, self.num_b_stage, self.num_epi_stage_per_tile = (2, 4, 4, 4)
            self.tmem_alloc_cols = self.num_acc_stage * self.mma_tiler[1]
            assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

            self.cta_tile_shape_mnk = (
                self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler[1],
                self.mma_tiler[2],
            )

        @cute.kernel
        def kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            mLabels: cute.Tensor,
            mMax: cute.Tensor,
            mAccu: cute.Tensor,
            mLogprobs: cute.Tensor,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            cluster_layout_vmnk: cute.Layout,
            problem_mnk: Tuple[int, int, int],
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            # FIXME: block swizzling applied here
            pidm, pidn = bidx, bidy

            # prefetch tma descriptors
            if warp_idx == self.load_warp_ids:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

            # declare SMEM
            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                num_stages=self.num_a_stage,
                producer_group=_thread_group(len([self.load_warp_ids])),
                consumer_group=_thread_group(len([self.mma_warp_ids])),
                tx_count=self.tma_copy_a_bytes + self.tma_copy_b_bytes,
                barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            )
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_a_stage)
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_a_stage)

            mma_pipeline = pipeline.PipelineUmmaAsync.create(
                num_stages=self.num_acc_stage,
                producer_group=_thread_group(len([self.mma_warp_ids])),
                consumer_group=_thread_group(self.threads_per_warp * len(self.epi_warp_ids)),
                barrier_storage=storage.mma_mbar_ptr.data_ptr(),
            )
            mma_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            mma_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
            if warp_idx == self.empty_warp_ids[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.threads_per_warp * len(self.epi_warp_ids))
                    cute.arch.mbarrier_init_fence()

            # -------- SMEM partition ------------ #
            # swizzle o [(tileM, tileK), loopM, loopK, Stage]
            sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
            # swizzle o [(tileN, tileK), loopN, loopK, stage]
            sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

            # FIXME: if 2 CTAs, modify here
            thr_mma = tiled_mma.get_slice(0)
            # [MMA, loopM, loopK, stage]
            tCsA = thr_mma.make_fragment_A(sA)
            # [MMA, loopN, loopK, stage]
            tCsB = thr_mma.make_fragment_B(sB)

            # ---------- GMEM partition ----------- #
            # [tileM, tileK, loopK]
            gA = cute.local_tile(mA, (self.mma_tiler[0], self.mma_tiler[2]), (pidm, None))

            # [vocab_size_per_split, dim]
            mB_n = cute.local_tile(mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (pidn, 0))

            # [tileN, tileK, loopN, loopK]
            gB = cute.local_tile(mB_n, (self.mma_tiler[1], self.mma_tiler[2]), (None, None))

            # [MMA, tileCntM, tileCntK, loopK]
            tCgA = thr_mma.partition_A(gA)
            # [MMA, tileCntN, tileCntK, loopN, loopK]
            tCgB = thr_mma.partition_B(gB)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            # FIXME: if 2 CTAs, modify here
            cta_rank_in_cluster = 0
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
            tTMAsA, tTMAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],  # cta_coord,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),  # SMEM tensor
                cute.group_modes(tCgA, 0, 3),  # GMEM tensor
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            tTMAsB, tTMAgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],  # cta_coord
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # Allocate TMEM
            tmem_holding_buf = storage.tmem_holding_buf
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_holding_buf, is_two_cta=self.use_2cta_instrs)
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )

            # [(tileM, tileN), loopM, loopN]
            tmem_shape = (128, self.tmem_alloc_cols)
            acc_shape = thr_mma.partition_shape_C(tmem_shape)
            tCtC_fake = thr_mma.make_fragment_C(acc_shape)
            tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

            block_vocab_left_idx: cutlass.Int64 = pidn * self.vocab_per_split
            block_vocab_right_idx: cutlass.Int64 = min((pidn + 1) * self.vocab_per_split, problem_mnk[1])
            num_n_tiles: cutlass.Int64 = cute.ceil_div(
                (block_vocab_right_idx - block_vocab_left_idx), self.mma_tiler[1]
            )

            # ///////
            # empty
            # ///////
            if warp_idx in self.empty_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # ///////
            # load
            # ///////
            if warp_idx == self.load_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for n in cutlass.range(num_n_tiles):
                    for k in cutlass.range(cute.size(gA, mode=[2])):
                        ab_pipeline.producer_acquire(ab_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tTMAgA[(None, k)],
                            tTMAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        )
                        cute.copy(
                            tma_atom_b,
                            tTMAgB[(None, n, k)],
                            tTMAsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        )
                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()

            # ///////
            # mma
            # ///////
            if warp_idx == self.mma_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for n in cutlass.range(num_n_tiles):
                    # disable accumulate for the first tile
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    mma_pipeline.producer_acquire(mma_producer_state)

                    for k in cutlass.range(cute.size(gA, mode=[2])):
                        ab_pipeline.consumer_wait(ab_consumer_state)

                        for kblock_idx in cutlass.range(cute.size(tCsA, mode=[2]), unroll_full=True):
                            cute.gemm(
                                tiled_mma,
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                                tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                                tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                                cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                            )
                            # enable accumulate for the next tile
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()

                    mma_pipeline.producer_commit(mma_producer_state)
                    mma_producer_state.advance()

            # //////////
            # epilogue
            # //////////
            if warp_idx in self.epi_warp_ids:
                cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

                # epilog TMEM copy and partition
                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    cutlass_utils.LayoutEnum.ROW_MAJOR,  # This is hard-coded
                    self.acc_dtype,
                    self.acc_dtype,
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                    self.use_2cta_instrs,
                )
                # [tileM, subTileN, loopM, CntSubTileN, loopN]
                tAcc_epi = cute.flat_divide(
                    tCtC[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTMEM_load_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                # [(pattern), loopM, loopN, CntTileM, CntTileN]
                tTMEM_load_tAcc = cute.group_modes(tTMEM_load_tAcc, 3, cute.rank(tTMEM_load_tAcc) - 1)

                cAcc = cute.make_identity_tensor(self.mma_tiler[:2])
                tCcAcc = thr_mma.partition_C(cAcc)
                # [tileM, subTileN, loopM, CntSubTileN, CntTileN]
                tCcAcc_epi = cute.flat_divide(
                    tCcAcc[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tTMEM_load_cAcc = thr_copy_t2r.partition_D(tCcAcc_epi)
                tTMEM_load_cAcc_shape = cute.select(tTMEM_load_cAcc.shape, mode=[0, 1, 2])

                # epilogue layouts
                epilogue_thread_layout = cute.make_layout((128, 1))
                copy_atom_g2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mLabels.element_type)
                tiled_copy_g2r = cute.make_tiled_copy(copy_atom_g2r, epilogue_thread_layout, (128, 1))
                thr_copy_g2r = tiled_copy_g2r.get_slice(tidx)

                copy_atom_r2g = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
                tiled_copy_r2g = cute.make_tiled_copy(copy_atom_r2g, epilogue_thread_layout, (128, 1))
                thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

                # auxiliary tensors
                # [tileM]
                gLabels = cute.local_tile(mLabels, (self.epi_tile[0],), (pidm,))

                tLabelsCAcc = thr_copy_g2r.partition_S(cAcc)[(None, None, 0)]
                tLabelsCAcc_mask = cute.make_fragment(tLabelsCAcc.shape, cutlass.Boolean)
                # [(1, 1), 1]
                tLabelsCAcc_mask[0] = cute.elem_less(pidm * self.epi_tile[0] + tidx, problem_mnk[0])
                # to align shape with gMax and gAccu
                tLabelsCAcc_mask = cute.append_ones(tLabelsCAcc_mask)

                # [(1, 1), 1, 1]
                tLabelsgLabels = thr_copy_g2r.partition_S(cute.append_ones(gLabels))
                tLabelsrLabels = cute.make_fragment(tLabelsgLabels.shape, tLabelsgLabels.element_type)
                cute.copy(tiled_copy_g2r, tLabelsgLabels, tLabelsrLabels, pred=tLabelsCAcc_mask)
                valid_mask: cutlass.Boolean = (tLabelsrLabels[0] != ignore_index) and tLabelsCAcc_mask[0]

                # [tileM, 1]
                gMax = cute.local_tile(mMax, (self.epi_tile[0], 1), (pidm, pidn))
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgMax = thr_copy_r2g.partition_D(gMax)
                tR2GrMax = cute.make_fragment(tR2GgMax.shape, tR2GgMax.element_type)
                tR2GrMax.fill(-1e30)

                # [tileM, 1]
                gAccu = cute.local_tile(mAccu, (self.epi_tile[0], 1), (pidm, pidn))
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgAccu = thr_copy_r2g.partition_D(gAccu)
                tR2GrAccu = cute.make_fragment(tR2GgAccu.shape, tR2GgAccu.element_type)
                tR2GrAccu.fill(0.0)

                # [tileM, 1]
                gLogprobs = cute.append_ones(cute.local_tile(mLogprobs, (self.epi_tile[0],), (pidm,)))
                # [(CPYM, CPYN), loopM, loopN]
                tR2GgLogprobs = thr_copy_r2g.partition_D(gLogprobs)
                tR2GrLogprobs = cute.make_fragment(tR2GgLogprobs.shape, tR2GgLogprobs.element_type)
                tR2GrLogprobs.fill(0.0)

                # [(tileN // num_epi_stage_per_tile, 1), 1, 1]
                tTMEM_load_rAcc = cute.make_fragment(tTMEM_load_cAcc_shape, self.acc_dtype)

                for n in cutlass.range(num_n_tiles):
                    mma_pipeline.consumer_wait(mma_consumer_state)

                    left: cutlass.Int64 = block_vocab_left_idx + n * self.epi_tile[1]
                    right: cutlass.Int64 = min(
                        (n + 1) * self.epi_tile[1] + block_vocab_left_idx, block_vocab_right_idx
                    )
                    num_n_subtiles: cutlass.Int64 = cute.ceil_div((right - left), cute.size(tTMEM_load_rAcc, mode=[0]))
                    for n_subtile in cutlass.range(num_n_subtiles):
                        cute.copy(
                            tiled_copy_t2r,
                            tTMEM_load_tAcc[(None, None, None, n_subtile, mma_consumer_state.index)],
                            tTMEM_load_rAcc,
                        )

                        for idx in cutlass.range(cute.size(tTMEM_load_rAcc, mode=[0]), unroll_full=True):
                            local_position: cutlass.Int64 = (
                                n * self.epi_tile[1] + n_subtile * cute.size(tTMEM_load_rAcc, mode=[0]) + idx
                            )
                            if (block_vocab_left_idx + local_position) < block_vocab_right_idx:
                                _max_old = tR2GrMax[0]
                                tR2GrMax[0] = cute.arch.fmax(tR2GrMax[0], tTMEM_load_rAcc[idx])
                                exp_logits = cute.exp(tTMEM_load_rAcc[idx] - tR2GrMax[0])
                                coeff = cute.exp(_max_old - tR2GrMax[0])
                                tR2GrAccu[0] = coeff * tR2GrAccu[0] + exp_logits

                                position: cutlass.Int64 = (
                                    rank * problem_mnk[1] + pidn * self.vocab_per_split + local_position
                                )
                                mask: cutlass.Boolean = valid_mask and (position == tLabelsrLabels[0])
                                tR2GrLogprobs[0] += mask * tTMEM_load_rAcc[idx]

                    mma_pipeline.consumer_release(mma_consumer_state)
                    mma_consumer_state.advance()

                cute.copy(tiled_copy_r2g, tR2GrMax, tR2GgMax, pred=tLabelsCAcc_mask)
                cute.copy(tiled_copy_r2g, tR2GrAccu, tR2GgAccu, pred=tLabelsCAcc_mask)

                vocab_left_idx: cutlass.Int64 = rank * problem_mnk[1] + pidn * self.vocab_per_split
                vocab_right_idx: cutlass.Int64 = rank * problem_mnk[1] + min(
                    (pidn + 1) * self.vocab_per_split, problem_mnk[1]
                )
                valid: cutlass.Boolean = tLabelsrLabels[0] >= vocab_left_idx and tLabelsrLabels[0] < vocab_right_idx
                tLabelsCAcc_mask[0] &= valid

                cute.copy(tiled_copy_r2g, tR2GrLogprobs, tR2GgLogprobs, pred=tLabelsCAcc_mask)

            # Dealloc TMEM
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols, is_two_cta=self.use_2cta_instrs)

        @staticmethod
        def _compute_grid(
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
            num_splits: int,
        ) -> Tuple[int, int, int]:

            cluster_shape = (*cluster_shape_mn, 1)

            grid = cute.round_up((cute.ceil_div(problem_mnk[0], cta_tiler[0]), num_splits, 1), cluster_shape)
            return grid

        @cute.jit
        def __call__(
            self,
            hidden: cute.Tensor,
            weight: cute.Tensor,
            labels: cute.Tensor,
            _logprobs: cute.Tensor,
            _max: cute.Tensor,
            _accu: cute.Tensor,
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
            stream: cuda.CUstream,
        ) -> None:
            a_dtype: Type[cutlass.Numeric] = hidden.element_type
            b_dtype: Type[cutlass.Numeric] = weight.element_type

            if cutlass.const_expr(hidden.element_type != weight.element_type):
                raise RuntimeError(f"data type don't match: {hidden.element_type} v.s. {weight.element_type}")
            if cutlass.const_expr(hidden.element_type not in [cutlass.Float16, cutlass.BFloat16]):
                raise RuntimeError("hidden can only be FP16 or BF16")
            if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
                raise RuntimeError("K dimension doesn't match")

            problem_mnk = (hidden.layout.shape[0], weight.layout.shape[0], hidden.layout.shape[1])
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
                raise RuntimeError(f"K dimension is not 16B aligned: {problem_mnk[2]}")

            num_splits = cute.ceil_div(problem_mnk[1], self.vocab_per_split)

            grid = self._compute_grid(
                problem_mnk=problem_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                cta_tiler=self.cta_tiler,
                num_splits=num_splits,
            )
            a_major_mode = cutlass_utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
            b_major_mode = cutlass_utils.LayoutEnum.from_tensor(weight).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype,
                a_major_mode,
                b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )

            self._setup_attributes(tiled_mma)
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 128 != 0):
                raise RuntimeError(f"K dimension is not 128B aligned: {problem_mnk[2]}")

            self.epi_tile = self.mma_tiler[:2]

            # Swizzle o [(tileM, tileK), loopM, loopK, stage]
            a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, a_dtype, self.num_a_stage)
            # Swizzle o [(tileN, tileK), loopN, loopK, stage]
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, self.num_b_stage)

            # TMA loading
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

            # Swizzle o [(tileM, tileK), loopM, loopK]
            a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
            # create tma copy atom for hidden,
            # and the cooresponding tma descriptor tensor
            tma_atom_a, tma_desc_a = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                hidden,  # gmem_tensor
                a_smem_layout,  # SMEM layout
                self.mma_tiler,  # MMA tiler
                tiled_mma,  # TiledMMA
                self.cluster_layout_vmnk.shape,  # cluster_shape_vmnk
            )
            # Swizzle o [(tileN, tileK), loopN, loopK]
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_b, tma_desc_b = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                weight,  # gmem_tensor
                b_smem_layout,  # SMEM layout
                self.mma_tiler,  # MMA tiler
                tiled_mma,  # TiledMMA
                self.cluster_layout_vmnk.shape,  # cluster_shape_vmnk
            )
            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.tma_copy_a_bytes = a_copy_size
            self.tma_copy_b_bytes = b_copy_size

            assert self.num_a_stage == self.num_b_stage

            @cute.struct
            class SharedStorage:
                # pipeline barriers, 2 = producer + consumer
                load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
                mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                # tmem holding buffer
                tmem_holding_buf: cutlass.Int32
                # SMEM tensors
                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage

            # launch kernel
            self.kernel(
                tiled_mma,
                tma_atom_a,
                tma_desc_a,
                tma_atom_b,
                tma_desc_b,
                labels,
                _max,
                _accu,
                _logprobs,
                a_smem_layout_staged,
                b_smem_layout_staged,
                self.cluster_layout_vmnk,
                problem_mnk,
                ignore_index,
                rank,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
            )
            return None

    # CuTe backward dLogits kernel.

    class BwdPartialDlogits:
        def __init__(self, vocab_per_split: int):
            self.acc_dtype = cutlass.Float32
            self.use_2cta_instrs = False
            self.mma_tiler = (128, 256, 1)
            self.vocab_per_split = vocab_per_split

            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)

            self.threads_per_warp: int = 32

            self.epi_warp_ids = (0, 1, 2, 3)
            self.load_warp_ids = 4
            self.mma_warp_ids = 5
            self.empty_warp_ids = (6, 7)

            self.threads_per_cta: int = self.threads_per_warp * len(
                (*self.epi_warp_ids, self.load_warp_ids, self.mma_warp_ids, *self.empty_warp_ids)
            )
            self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)

            self.buffer_align_bytes: int = 1024
            self.num_regs_other: int = 32
            self.num_regs_epi: int = 192

        def _compute_grid(
            self,
            problem_mnk: Tuple[int, int, int],
            cluster_shape_mn: Tuple[int, int],
            cta_tiler: Tuple[int, int, int],
        ) -> Tuple[int, int, int]:
            cluster_shape_mnk = (*cluster_shape_mn, 1)

            grid = cute.round_up(
                (
                    cute.ceil_div(problem_mnk[0], cta_tiler[0]),
                    cute.ceil_div(self.vocab_per_split, cta_tiler[1]),
                    1,
                ),
                cluster_shape_mnk,
            )
            return grid

        def _setup_attributes(self, tiled_mma: cute.TiledMma):
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk), (tiled_mma.thr_id.shape,)
            )

            mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            # it requires k-mode to be 128B aligned
            mma_inst_tile_k: int = 4
            self.mma_tiler = (
                self.mma_tiler[0],
                self.mma_tiler[1],
                mma_inst_shape_k * mma_inst_tile_k,
            )

            self.num_acc_stage, self.num_ab_stage, self.num_epi_stage_per_tile = (1, 4, 4)
            self.tmem_alloc_cols = self.num_acc_stage * self.mma_tiler[1]
            assert self.tmem_alloc_cols <= SM100_TMEM_CAPACITY_COLUMNS

            self.cta_tile_shape_mnk = (
                self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
                self.mma_tiler[1],
                self.mma_tiler[2],
            )

        @cute.kernel
        def kernel(
            self,
            split_idx: cutlass.Int32,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            mA: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            mB: cute.Tensor,
            mLabels: cute.Tensor,
            mDlogprobs: cute.Tensor,
            mMaximum: cute.Tensor,
            mAccu: cute.Tensor,
            mDlogits_partial: cute.Tensor,
            ignore_index: cutlass.Int64,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            cluster_layout_vmnk: cute.Layout,
            problem_mnk: Tuple[int, int, int],
            rank: cutlass.Int32,
        ) -> None:
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            # FIXME: block swizzling applied here
            pidm, pidn = bidx, bidy

            # FIXME: if 2 CTAs, modify here
            cta_rank_in_cluster = 0
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

            # prefetch tma descriptors
            if warp_idx == self.load_warp_ids:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                num_stages=self.num_ab_stage,
                producer_group=_thread_group(len([self.load_warp_ids])),
                consumer_group=_thread_group(len([self.mma_warp_ids])),
                tx_count=self.tma_copy_ab_bytes,
                barrier_storage=storage.load_ab_mbar_ptr.data_ptr(),
            )
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)

            mma_pipeline = pipeline.PipelineUmmaAsync.create(
                num_stages=self.num_acc_stage,
                producer_group=_thread_group(len([self.mma_warp_ids])),
                consumer_group=_thread_group(self.threads_per_warp * len(self.epi_warp_ids)),
                barrier_storage=storage.mma_mbar_ptr.data_ptr(),
            )
            mma_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            mma_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
            if warp_idx == self.empty_warp_ids[0]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.threads_per_warp * len(self.epi_warp_ids))
                    cute.arch.mbarrier_init_fence()

            # -------- tensor partition ------------ #
            # swizzle o [(tileM, tileK), loopM, loopK, stage]
            sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
            # swizzle o [(tileN, tileK), loopN, loopK, stage]
            sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

            # FIXME: if 2 CTAs, modify here
            thr_mma = tiled_mma.get_slice(0)
            # [MMA, loopM, loopK, stage]
            tCsA = thr_mma.make_fragment_A(sA)
            # [MMA, loopN, loopK, stage]
            tCsB = thr_mma.make_fragment_B(sB)

            # [tileM, tileK, loopK]
            gA = cute.local_tile(mA, (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]), (pidm, None))
            # [vocab_per_split, dim]
            mB_n = cute.local_tile(mB, (self.vocab_per_split, cute.size(mB.layout.shape, mode=[1])), (split_idx, 0))
            # [tileN, tileK, loopK]
            gB = cute.local_tile(mB_n, (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]), (pidn, None))

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            # just to make sure SMEM and GMEM tensor has the same size in the first rank
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)
            # [CPY, stage] & [CPY, loopK]
            tTMAsA, tTMAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],  # cta_coord,
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            # [CPY, stage] & [CPY, loopK]
            tTMAsB, tTMAgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],  # cta_coord
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # ------ Allocate TMEM ------ #
            tmem_holding_buf = storage.tmem_holding_buf
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.alloc_tmem(self.tmem_alloc_cols, tmem_holding_buf, is_two_cta=self.use_2cta_instrs)
            self.cta_sync_barrier.arrive_and_wait()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )

            tmem_shape = (128, self.tmem_alloc_cols)
            acc_shape = thr_mma.partition_shape_C(tmem_shape)
            tCtC_fake = thr_mma.make_fragment_C(acc_shape)
            # [(tileM, tileN), loopM, loopN]
            tCtC = cute.make_tensor(tmem_ptr, tCtC_fake.layout)

            # ------ Empty ------ #
            if warp_idx in self.empty_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # ------ Load ------ #
            if warp_idx == self.load_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tTMAgA[(None, k)],
                        tTMAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tTMAgB[(None, k)],
                        tTMAsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

            # ------ MMA ------ #
            if warp_idx == self.mma_warp_ids:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                mma_pipeline.producer_acquire(mma_producer_state)

                for k in cutlass.range(cute.size(gA, mode=[2])):
                    ab_pipeline.consumer_wait(ab_consumer_state)

                    for kblock_idx in cutlass.range(cute.size(tCsA, mode=[2]), unroll_full=True):
                        cute.gemm(
                            tiled_mma,
                            cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                            tCsA[(None, None, kblock_idx, ab_consumer_state.index)],
                            tCsB[(None, None, kblock_idx, ab_consumer_state.index)],
                            cute.append_ones(tCtC[(None, None, mma_producer_state.index)]),
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()

                mma_pipeline.producer_commit(mma_producer_state)
                mma_producer_state.advance()

            # ------ EPI ------ #
            if warp_idx in self.epi_warp_ids:
                cute.arch.warpgroup_reg_alloc(self.num_regs_epi)

                copy_atom_t2r = sm100_utils.get_tmem_load_op(
                    self.cta_tile_shape_mnk,
                    cutlass_utils.LayoutEnum.ROW_MAJOR,
                    self.acc_dtype,
                    self.acc_dtype,
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                    self.use_2cta_instrs,
                )
                # [tileM, subTileN, loopM, CntSubTileN, loopN]
                tAcc_epi = cute.flat_divide(
                    tCtC[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                tTMEM_load_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                tTMEM_load_tAcc = cute.group_modes(tTMEM_load_tAcc, 3, cute.rank(tTMEM_load_tAcc) - 1)

                # predicates
                cAcc = cute.make_identity_tensor(self.mma_tiler[:2])
                tCcAcc = thr_mma.partition_C(cAcc)
                tCcAcc_epi = cute.flat_divide(
                    tCcAcc[((None, None), 0, None)],
                    (self.epi_tile[0], self.epi_tile[1] // self.num_epi_stage_per_tile),
                )
                tTMEM_load_cAcc = thr_copy_t2r.partition_D(tCcAcc_epi)
                tTMEM_load_cAcc_shape = cute.select(tTMEM_load_cAcc.shape, mode=[0, 1, 2])
                tTMEM_load_rAcc = cute.make_fragment(tTMEM_load_cAcc_shape, self.acc_dtype)

                copy_atom_g2r_int64 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mLabels.element_type)
                copy_atom_g2r_fp32 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mDlogprobs.element_type)
                epilogue_thread_layout = cute.make_layout((128, 1), stride=(1, 1))
                tiled_copy_g2r_int64 = cute.make_tiled_copy_tv(
                    copy_atom_g2r_int64, epilogue_thread_layout, cute.make_layout((1, 1))
                )
                tiled_copy_g2r_fp32 = cute.make_tiled_copy_tv(
                    copy_atom_g2r_fp32, epilogue_thread_layout, cute.make_layout((1, 1))
                )
                thr_copy_g2r_int64 = tiled_copy_g2r_int64.get_slice(tidx)
                thr_copy_g2r_fp32 = tiled_copy_g2r_fp32.get_slice(tidx)

                # [tileM]
                gLabels = cute.local_tile(mLabels, (self.epi_tile[0],), (pidm,))
                gMaximum = cute.local_tile(mMaximum, (self.epi_tile[0],), (pidm,))
                gAccu = cute.local_tile(mAccu, (self.epi_tile[0],), (pidm,))

                # slice along M direction
                tMCAcc = thr_copy_g2r_int64.partition_S(cAcc)[(None, None, 0)]
                # [(1, 1), 1]
                tMCAcc_mask = cute.make_fragment(tMCAcc.shape, cutlass.Boolean)
                # to align shape with gMax and gAccu
                tMCAcc_mask = cute.append_ones(tMCAcc_mask)
                tMCAcc_mask[0] = cute.elem_less(pidm * self.epi_tile[0] + tidx, cute.size(mA, mode=[0]))
                # [(1, 1), 1, 1]
                tMgLabels = thr_copy_g2r_int64.partition_S(cute.append_ones(gLabels))
                tMrLabels = cute.make_fragment(tMgLabels.shape, tMgLabels.element_type)
                cute.copy(tiled_copy_g2r_int64, tMgLabels, tMrLabels, pred=tMCAcc_mask)
                tMgMaximum = thr_copy_g2r_fp32.partition_S(cute.append_ones(gMaximum))
                tMrMaximum = cute.make_fragment(tMgMaximum.layout, tMgMaximum.element_type)
                cute.copy(tiled_copy_g2r_fp32, tMgMaximum, tMrMaximum, pred=tMCAcc_mask)
                tMgAccu = thr_copy_g2r_fp32.partition_S(cute.append_ones(gAccu))
                tMrAccu = cute.make_fragment(tMgAccu.layout, tMgAccu.element_type)
                cute.copy(tiled_copy_g2r_fp32, tMgAccu, tMrAccu, pred=tMCAcc_mask)

                tMrDlogprobs = cute.make_fragment(tMgAccu.layout, mDlogprobs.element_type)
                gDlogprobs = cute.local_tile(mDlogprobs, (self.epi_tile[0],), (pidm,))
                tMgDlogprobs = thr_copy_g2r_fp32.partition_S(cute.append_ones(gDlogprobs))
                cute.copy(tiled_copy_g2r_fp32, tMgDlogprobs, tMrDlogprobs, pred=tMCAcc_mask)

                tMrAccu[0] = cute.arch.rcp_approx(tMrAccu[0])
                tMrDlogprobs[0] *= tMrLabels[0] != ignore_index
                tMr_d_acc_exp_logits = tMrDlogprobs[0] * tMrAccu[0]

                # ------ Partial output ------ #
                # [tileM, tileN]
                gDlogits_partial = cute.local_tile(
                    mDlogits_partial, (self.epi_tile[0], self.epi_tile[1]), (cutlass.Int64(pidm), cutlass.Int64(pidn))
                )
                # blackwell supports STG.256
                copy_atom_r2g = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    gDlogits_partial.element_type,
                    num_bits_per_copy=256,
                )
                tiled_copy_r2g = cute.make_tiled_copy_tv(
                    copy_atom_r2g, epilogue_thread_layout, copy_atom_r2g.layout_dst_tv
                )
                thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

                # [CPY, loopM, loopN]
                tR2GCAcc = thr_copy_r2g.partition_S(cAcc)
                tR2GCAcc_pred = cute.make_fragment(tR2GCAcc.shape, cutlass.Boolean)
                for elem in cutlass.range(cute.size(tR2GCAcc_pred, mode=[0])):
                    for row in cutlass.range(cute.size(tR2GCAcc_pred, mode=[1])):
                        for col in cutlass.range(cute.size(tR2GCAcc_pred, mode=[2])):
                            tR2GCAcc_pred[elem, row, col] = cute.elem_less(
                                pidm * self.epi_tile[0] + tR2GCAcc[elem, row, col][0],
                                problem_mnk[0],
                            ) and cute.elem_less(
                                split_idx * self.vocab_per_split
                                + pidn * self.epi_tile[1]
                                + tR2GCAcc[elem, row, col][1],
                                problem_mnk[1],
                            )

                tR2GgDlogits = thr_copy_r2g.partition_D(gDlogits_partial)

                # for type conversion
                dLogits_half = cute.make_fragment(tTMEM_load_rAcc.shape, tR2GgDlogits.element_type)
                dLogits_half = cute.tiled_divide(dLogits_half, (cute.size(tR2GgDlogits, mode=[0]), 1))
                dLogits_half = cute.group_modes(dLogits_half, 2, cute.rank(dLogits_half))

                mma_pipeline.consumer_wait(mma_consumer_state)

                block_vocab_left_idx: cutlass.Int64 = split_idx * self.vocab_per_split + pidn * self.epi_tile[1]
                block_vocab_right_idx: cutlass.Int64 = min(
                    split_idx * self.vocab_per_split + (pidn + 1) * self.epi_tile[1],
                    min((split_idx + 1) * self.vocab_per_split, problem_mnk[1]),
                )
                num_n_subtiles: cutlass.Int64 = cute.ceil_div(
                    (block_vocab_right_idx - block_vocab_left_idx),
                    cute.size(tTMEM_load_rAcc, mode=[0]),
                )
                for n_subtile in cutlass.range(num_n_subtiles):
                    cute.copy(
                        tiled_copy_t2r,
                        tTMEM_load_tAcc[(None, None, None, n_subtile, mma_consumer_state.index)],
                        tTMEM_load_rAcc,
                    )

                    for idx in cutlass.range(cute.size(tTMEM_load_rAcc, mode=[0]), unroll_full=True):
                        # exp_logits
                        tTMEM_load_rAcc[idx] = cute.exp(tTMEM_load_rAcc[idx] - tMrMaximum[0])

                        position: cutlass.Int64 = (
                            rank * problem_mnk[1]
                            + split_idx * self.vocab_per_split
                            + pidn * self.epi_tile[1]
                            + n_subtile * cute.size(tTMEM_load_rAcc, mode=[0])
                            + idx
                        )
                        mask: cutlass.Boolean = position == tMrLabels[0] and tMrLabels[0] != ignore_index
                        # d_logits
                        tTMEM_load_rAcc[idx] *= tMr_d_acc_exp_logits
                        tTMEM_load_rAcc[idx] += mask * -tMrDlogprobs[0]
                        dLogits_half[idx] = tTMEM_load_rAcc[idx].to(dLogits_half.element_type)

                    for idx in cutlass.range(cute.size(dLogits_half, mode=[1]), unroll_full=True):
                        copy_id = n_subtile * cute.size(dLogits_half, mode=[1]) + idx
                        cute.copy(
                            tiled_copy_r2g,
                            dLogits_half[(None, idx, None)],
                            tR2GgDlogits[(None, None, copy_id)],
                            pred=tR2GCAcc_pred[((0, None), None, copy_id)],
                        )

                mma_pipeline.consumer_release(mma_consumer_state)
                mma_consumer_state.advance()

            # ------ Deallocate TMEM ------ #
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols, is_two_cta=self.use_2cta_instrs)

        @cute.jit
        def __call__(
            self,
            split_idx: cutlass.Int32,
            hidden: cute.Tensor,
            weight: cute.Tensor,
            labels: cute.Tensor,
            dlogprobs: cute.Tensor,
            maximum: cute.Tensor,
            accu: cute.Tensor,
            dlogits_partial: cute.Tensor,
            ignore_index: cutlass.Int64,
            rank: cutlass.Int32,
            stream: cuda.CUstream,
        ) -> None:
            a_dtype: Type[cutlass.Numeric] = hidden.element_type
            b_dtype: Type[cutlass.Numeric] = weight.element_type

            if cutlass.const_expr(hidden.element_type != weight.element_type):
                raise RuntimeError(f"data type don't match: {hidden.element_type} v.s. {weight.element_type}")
            if cutlass.const_expr(hidden.element_type not in [cutlass.Float16, cutlass.BFloat16]):
                raise RuntimeError("hidden can only be FP16 or BF16")
            if cutlass.const_expr(hidden.layout.shape[1] != weight.layout.shape[1]):
                raise RuntimeError("K dimension doesn't match")

            problem_mnk = (hidden.layout.shape[0], weight.layout.shape[0], hidden.layout.shape[1])
            if cutlass.const_expr((problem_mnk[2] * a_dtype.width // 8) % 16 != 0):
                raise RuntimeError(f"K dimension is not 16B aligned: {problem_mnk[2]}")
            if cutlass.const_expr((problem_mnk[2] * b_dtype.width // 8) % 128 != 0):
                raise RuntimeError(f"N dimension is not 128B aligned: {problem_mnk[1]}")

            grid = self._compute_grid(
                problem_mnk=problem_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                cta_tiler=self.mma_tiler,
            )

            a_major_mode = cutlass_utils.LayoutEnum.from_tensor(hidden).mma_major_mode()
            b_major_mode = cutlass_utils.LayoutEnum.from_tensor(weight).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype,
                a_major_mode,
                b_major_mode,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler[:2],
            )
            self._setup_attributes(tiled_mma)

            self.epi_tile = self.cta_tile_shape_mnk[:2]

            # Swizzle o [(tileM, tileK), loopM, loopK, stage]
            a_smem_layout_staged = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, a_dtype, self.num_ab_stage
            )
            # Swizzle o [(tileN, tileK), loopN, loopK, stage]
            b_smem_layout_staged = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, b_dtype, self.num_ab_stage
            )
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

            # Swizzle o [(tileM, tileK), loopM, loopK]
            a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                hidden,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            # Swizzle o [(tileN, tileK), loopN, loopK]
            b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])
            tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                weight,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            a_copy_size = cute.size_in_bytes(a_dtype, a_smem_layout)
            b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
            self.tma_copy_ab_bytes = a_copy_size + b_copy_size

            @cute.struct
            class SharedStorage:
                load_ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
                mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]

                tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
                tmem_holding_buf: cutlass.Int32

                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]

            self.shared_storage = SharedStorage

            self.kernel(
                split_idx,
                tiled_mma,
                tma_atom_a,
                tma_tensor_a,
                tma_atom_b,
                tma_tensor_b,
                labels,
                dlogprobs,
                maximum,
                accu,
                dlogits_partial,
                ignore_index,
                a_smem_layout_staged,
                b_smem_layout_staged,
                self.cluster_layout_vmnk,
                problem_mnk,
                rank,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
            )

    # Host dispatch and cached kernel configurations.
    @dataclass
    class FwdConfig:
        _dedicated_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)
        _dedicated_events: typing.List[torch.cuda.Event] = field(default_factory=list)
        _initialized: bool = field(default=False)
        _fwd_mainloop_kernels: typing.Dict[str, cute.kernel] = field(default_factory=dict)
        _vocab_per_split: int = field(default=int(os.environ.get("LCE_FWD_VOCAB_SPLIT_SIZE", 512 * 6)))

    @dataclass
    class BwdConfig:
        _bwd_kernel: typing.Dict[str, cute.kernel] = field(default_factory=dict)
        _vocab_per_split: int = field(default=int(os.environ.get("LCE_BWD_VOCAB_SPLIT_SIZE", 512 * 6)))

    @lru_cache(maxsize=1)
    def _get_fwd_config() -> FwdConfig:
        return FwdConfig()

    @lru_cache(maxsize=1)
    def _get_bwd_config() -> BwdConfig:
        return BwdConfig()

    def _blackwell_forward(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        tp_group: typing.Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        ignore_index = -100
        tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
        tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)
        in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

        assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
        assert weight.device == hidden.device and labels.device == hidden.device

        assert hidden.ndim == weight.ndim == 2 and labels.ndim == 1
        assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

        hidden_view = hidden.view(-1, hidden.shape[-1])
        labels_view = labels.view(-1)

        assert hidden_view.shape[0] == labels_view.shape[0]
        assert hidden_view.shape[1] == weight.shape[1]

        num_tokens, dim = hidden_view.shape
        vocab_size, _ = weight.shape

        if not _get_fwd_config()._initialized:
            _get_fwd_config()._dedicated_stream = torch.cuda.Stream(hidden.device)
            _get_fwd_config()._dedicated_events = [torch.cuda.Event() for _ in range(2)]
            _get_fwd_config()._initialized = True

        # declare logprobs
        logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        if in_tp_mode:
            logprobs.zero_()
        # declare auxiliary tensors
        maximum = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
        accumulate = torch.empty_like(maximum, dtype=torch.float32)
        assert maximum.is_contiguous() and accumulate.is_contiguous()
        # declare intermediate tensors
        # NOTE: this is a parameter for tuning
        num_splits = (vocab_size + _get_fwd_config()._vocab_per_split - 1) // _get_fwd_config()._vocab_per_split
        _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
        _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
        _logprobs = logprobs
        assert _max.is_contiguous() and _accu.is_contiguous() and _logprobs.is_contiguous()

        # need to compile the kernel for the first time
        hidden_packed = from_dlpack(hidden_view.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
        weight_packed = from_dlpack(weight.detach(), assumed_align=16)
        labels_packed = from_dlpack(labels_view.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
        logprobs_packed = from_dlpack(_logprobs, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        _max_packed = from_dlpack(_max, assumed_align=8).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        _accu_packed = from_dlpack(_accu, assumed_align=8).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        cuda_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # VocabSize and Dim are fixed for a given model,
        # only the number of tokens can vary
        key = f"vocab_size:{vocab_size}+dim:{dim}+dtype:{hidden_view.dtype}"
        if _get_fwd_config()._fwd_mainloop_kernels.get(key) is None:
            fwd_mainloop_kernel = FwdMainLoop(vocab_per_split=_get_fwd_config()._vocab_per_split)
            fwd_mainloop_compiled_kernel = cute.compile(
                fwd_mainloop_kernel,
                hidden_packed,
                weight_packed,
                labels_packed,
                logprobs_packed,
                _max_packed,
                _accu_packed,
                ignore_index,
                tp_rank,
                cuda_stream,
            )
            _get_fwd_config()._fwd_mainloop_kernels[key] = fwd_mainloop_compiled_kernel
        else:
            fwd_mainloop_compiled_kernel = _get_fwd_config()._fwd_mainloop_kernels[key]
        fwd_mainloop_compiled_kernel(
            hidden_packed,
            weight_packed,
            labels_packed,
            logprobs_packed,
            _max_packed,
            _accu_packed,
            ignore_index,
            tp_rank,
            cuda_stream,
        )

        if not in_tp_mode:

            def grid(meta):
                return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

            forward_dp_epilogue[grid](
                num_tokens,
                num_splits,
                ignore_index,
                labels_view,
                labels_view.stride(0),
                _max,
                _max.stride(0),
                _max.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                maximum,
                maximum.stride(0),
                accumulate,
                maximum.stride(0),
                _logprobs,
                _logprobs.stride(0),
            )
        else:
            _max_backup = _max.clone()
            dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=tp_group)

            torch.cuda.current_stream().record_event(_get_fwd_config()._dedicated_events[0])
            with torch.cuda.stream(_get_fwd_config()._dedicated_stream):
                _get_fwd_config()._dedicated_stream.wait_event(_get_fwd_config()._dedicated_events[0])
                dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=tp_group)
                _get_fwd_config()._dedicated_stream.record_event(_get_fwd_config()._dedicated_events[1])

            def grid(meta):
                return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

            forward_tp_epilogue[grid](
                num_tokens,
                num_splits,
                _max,
                _max.stride(0),
                _max.stride(1),
                _max_backup,
                _max_backup.stride(0),
                _max_backup.stride(1),
                _accu,
                _accu.stride(0),
                _accu.stride(1),
                maximum,
                maximum.stride(0),
                accumulate,
                maximum.stride(0),
            )
            # reduce accumulate
            dist.all_reduce(accumulate, op=dist.ReduceOp.SUM, group=tp_group)

            # update logprobs
            torch.cuda.current_stream().wait_event(_get_fwd_config()._dedicated_events[1])
            forward_tp_epilogue_update_logprobs[grid](
                num_tokens,
                ignore_index,
                labels_view,
                labels_view.stride(0),
                _logprobs,
                _logprobs.stride(0),
                maximum,
                maximum.stride(0),
                accumulate,
                accumulate.stride(0),
            )

        return logprobs, maximum, accumulate, tp_rank, tp_world_size

    def _blackwell_backward(
        dlogprobs: torch.Tensor,
        global_hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        maximum: torch.Tensor,
        accu: torch.Tensor,
        tp_group: typing.Optional[dist.ProcessGroup] = None,
        tp_rank: int = 0,
        tp_world_size: int = 1,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        ignore_index = -100
        in_tp_mode = (tp_group is not None) and (tp_world_size > 1)

        hidden_view = global_hidden.view(-1, global_hidden.shape[-1])
        labels_view = labels.view(-1)

        num_tokens, dim = hidden_view.shape
        vocab_size, _ = weight.shape

        dlogprobs_view = dlogprobs.view(-1)
        assert dlogprobs.shape == (num_tokens,)
        assert dlogprobs.is_contiguous() and dlogprobs.is_cuda

        # Allocate d_hidden in float32 for better numerical stability
        d_hidden = torch.empty_like(global_hidden, dtype=torch.float32)
        d_weight = torch.empty_like(weight)
        assert d_hidden.is_contiguous() and d_weight.is_contiguous()

        vocab_per_split = _get_bwd_config()._vocab_per_split
        num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

        _d_logits = torch.empty(
            (num_tokens, vocab_per_split),
            device=global_hidden.device,
            dtype=global_hidden.dtype,
        )

        hidden_packed = from_dlpack(hidden_view.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
        weight_packed = from_dlpack(weight.detach(), assumed_align=16)
        labels_packed = from_dlpack(labels_view.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
        dlogprobs_packed = from_dlpack(dlogprobs_view.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
        maximum_packed = from_dlpack(maximum.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
        accu_packed = from_dlpack(accu.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
        dlogits_packed = from_dlpack(_d_logits, assumed_align=32).mark_compact_shape_dynamic(mode=0)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        key = f"vocab_size:{vocab_size}+dim:{dim}+dtype:{hidden_view.dtype}"
        if _get_bwd_config()._bwd_kernel.get(key) is None:
            bwd_kernel = BwdPartialDlogits(vocab_per_split=vocab_per_split)
            bwd_kernel_compiled = cute.compile(
                bwd_kernel,
                0,  # split_idx
                hidden_packed,
                weight_packed,
                labels_packed,
                dlogprobs_packed,
                maximum_packed,
                accu_packed,
                dlogits_packed,
                ignore_index,
                tp_rank,
                stream,
            )
            _get_bwd_config()._bwd_kernel[key] = bwd_kernel_compiled
        else:
            bwd_kernel_compiled = _get_bwd_config()._bwd_kernel.get(key)

        for split_idx in range(num_splits):
            bwd_kernel_compiled(
                split_idx,
                hidden_packed,
                weight_packed,
                labels_packed,
                dlogprobs_packed,
                maximum_packed,
                accu_packed,
                dlogits_packed,
                ignore_index,
                tp_rank,
                stream,
            )
            # remove padding areas
            # cublas can handle non-contiguous tensors
            # therefore, we do not need to contiguous the tensor
            vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size) - split_idx * vocab_per_split
            valid_d_logits = _d_logits[:, :vocab_right_bound]

            _delta_hidden = torch.mm(
                valid_d_logits,
                weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :],
                out_dtype=torch.float32,
            ).view_as(d_hidden)
            if split_idx == 0:
                d_hidden.copy_(_delta_hidden)
            else:
                d_hidden.add_(_delta_hidden)
            torch.matmul(
                valid_d_logits.T,
                hidden_view,
                out=d_weight[split_idx * vocab_per_split : (split_idx + 1) * vocab_per_split, :],
            )

        if in_tp_mode:
            dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=tp_group)

        # convert d_hidden to the original dtype
        d_hidden = d_hidden.type_as(global_hidden)

        return d_hidden, d_weight
except ImportError as error:
    _KERNEL_IMPORT_ERROR = error
    get_logger(__name__).warning("Blackwell linear CE dependencies are unavailable: %s", error)


class _LinearCrossEntropy(torch.autograd.Function):
    """Per-token CE for already-gathered SP rows; retain Miles TP reduction
    order."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, tp_group):
        if _KERNEL_IMPORT_ERROR is not None:
            raise ImportError("Blackwell linear CE requires CUDA bindings, CuTe and Triton") from _KERNEL_IMPORT_ERROR
        if not hidden.is_cuda or torch.cuda.get_device_capability(hidden.device)[0] != 10:
            raise ValueError("Fused linear CE requires an SM100 GPU")
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            loss, maximum, accu, ctx.tp_rank, ctx.tp_size = _blackwell_forward(hidden, weight, labels, tp_group)
            ctx.save_for_backward(hidden, weight, labels, maximum, accu)
            ctx.tp_group = tp_group
        return loss

    @staticmethod
    def backward(ctx, dloss: torch.Tensor):
        hidden, weight, labels, maximum, accu = ctx.saved_tensors
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            dhidden, dweight = _blackwell_backward(
                dloss, hidden, weight, labels, maximum, accu, ctx.tp_group, ctx.tp_rank, ctx.tp_size
            )
        return dhidden, dweight, None, None


class _AccumulateWeightGradient(torch.autograd.Function):
    """Keep the original Parameter's DDP hook and FP32 main_grad contract."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.weight_ref = weight
        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return None
        weight = ctx.weight_ref
        # Miles returns a BF16 dWeight; preserve that kernel result and accumulate
        # it into the existing FP32 buffer, including contributions from MTP.
        weight.main_grad.add_(grad)
        if hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True
            return torch.zeros_like(weight) if getattr(weight, "zero_out_wgrad", False) else torch.empty_like(weight)
        return None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    tp_group: Any,
    *,
    accumulate: bool,
    return_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from megatron.core.utils import ensure_params_ready

    if hidden.numel() == 0:
        raise ValueError("Linear CE expects at least one row")
    if accumulate and weight.requires_grad:
        if not hasattr(weight, "main_grad"):
            raise ValueError("Fused gradient accumulation requires the original head Parameter.main_grad")
        if weight.main_grad.dtype != torch.float32:
            raise ValueError("Fused linear CE requires an FP32 main_grad buffer")
    ensure_params_ready([weight])
    x = hidden.reshape(-1, hidden.shape[-1]).to(weight.dtype).contiguous()
    labels = labels.reshape(-1).contiguous()
    kernel_weight = _AccumulateWeightGradient.apply(weight) if accumulate and weight.requires_grad else weight
    losses = _LinearCrossEntropy.apply(x, kernel_weight, labels, tp_group)
    # The caller has already gathered SP rows. Miles sums TP dHidden itself.
    logits = None
    if return_logits:
        # Existing MTP acceptance metrics consume local BF16 logits, without grad.
        with torch.no_grad():
            logits = x @ weight.T
    return losses.reshape(hidden.shape[:-1]), logits
