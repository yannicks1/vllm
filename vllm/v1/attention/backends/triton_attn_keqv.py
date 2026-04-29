# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend for Gemma4 global attention layers (k_eq_v).

Saves ~50% KV cache memory on global layers by storing only V in the
persistent per-layer cache. K is reconstructed from V before the standard
attention kernel runs, using the algebraic identity:

    K = RoPE(V * k_norm_weight)

This works because both k_norm and v_norm are RMS norms applied to the same
k_raw input (shared weights due to k_eq_v), and v_norm has no learnable
weight. The RMS terms cancel: K_pre_rope = V * k_norm_weight.

A lightweight Triton kernel reconstructs K for ALL positions into a shared
scratch buffer before each layer's attention. The attention kernel itself
is completely standard (unmodified) — no performance penalty.

Memory layout:
- Persistent cache: stores V only (50% savings vs K+V)
- Scratch buffer: one layer's worth of K, shared across all k_eq_v layers
  (layers execute sequentially, so only one is live at a time)
- Memory reservation: 1GB allocated during model init (before KV cache
  profiling) so the profiler sees reduced available memory. Freed at runtime
  when the real scratch is allocated.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    triton_reshape_and_cache_flash,
    unified_attention,
)


# --------------------------------------------------------------------------
# Reconstruction kernel: V → K for all cached positions
# --------------------------------------------------------------------------

@triton.jit
def _reconstruct_k_from_v_kernel(
    v_cache_ptr,
    k_scratch_ptr,
    k_norm_weight_ptr,
    cos_sin_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    block_table_stride: tl.int64,
    cache_stride_blk: tl.int64,
    cache_stride_slot: tl.int64,
    cache_stride_head: tl.int64,
    cos_sin_stride: tl.int64,
    max_blocks_per_seq: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    HALF_HEAD: tl.constexpr,
):
    """Reconstruct K = NeoX_RoPE(V * k_norm_weight) for one (seq_block, head).

    Grid: (num_seqs * max_blocks_per_seq, num_kv_heads)
    Each program reconstructs K for all slots in one physical block.
    """
    work_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_idx = work_idx // max_blocks_per_seq
    block_idx = work_idx % max_blocks_per_seq

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if block_idx * BLOCK_SIZE >= seq_len:
        return

    phys_block = tl.load(
        block_table_ptr + seq_idx * block_table_stride + block_idx
    ).to(tl.int64)

    # Slot positions and mask
    offs_slot = tl.arange(0, BLOCK_SIZE)
    positions = block_idx * BLOCK_SIZE + offs_slot
    slot_mask = positions < seq_len

    # Load k_norm_weight for both halves
    offs_half = tl.arange(0, HALF_HEAD)
    w_first = tl.load(k_norm_weight_ptr + offs_half).to(tl.float32)
    w_second = tl.load(k_norm_weight_ptr + HALF_HEAD + offs_half).to(tl.float32)

    # Base offset for this (physical_block, head)
    base_off = phys_block * cache_stride_blk + head_idx * cache_stride_head

    # Load V first half: [BLOCK_SIZE, HALF_HEAD]
    v_first_off = (
        base_off
        + offs_slot[:, None] * cache_stride_slot
        + offs_half[None, :]
    )
    V_first_raw = tl.load(
        v_cache_ptr + v_first_off, mask=slot_mask[:, None], other=0.0
    )
    V_first = V_first_raw.to(tl.float32)

    # Load V second half: [BLOCK_SIZE, HALF_HEAD]
    v_second_off = (
        base_off
        + offs_slot[:, None] * cache_stride_slot
        + (HALF_HEAD + offs_half)[None, :]
    )
    V_second = tl.load(
        v_cache_ptr + v_second_off, mask=slot_mask[:, None], other=0.0
    ).to(tl.float32)

    # K_pre = V * k_norm_weight (per half)
    Kp_first = V_first * w_first[None, :]
    Kp_second = V_second * w_second[None, :]

    # Load cos/sin for all positions in this block: [BLOCK_SIZE, HALF_HEAD]
    # cos_sin_cache layout: [max_pos, HEAD_SIZE] where first HALF_HEAD = cos,
    # second HALF_HEAD = sin.  Non-rotated dims are padded (cos=1, sin=0).
    cos_off = positions[:, None] * cos_sin_stride + offs_half[None, :]
    sin_off = positions[:, None] * cos_sin_stride + (HALF_HEAD + offs_half)[None, :]
    cos_vals = tl.load(
        cos_sin_cache_ptr + cos_off, mask=slot_mask[:, None], other=1.0
    ).to(tl.float32)
    sin_vals = tl.load(
        cos_sin_cache_ptr + sin_off, mask=slot_mask[:, None], other=0.0
    ).to(tl.float32)

    # NeoX rotation:
    #   K_first  = Kp_first * cos - Kp_second * sin
    #   K_second = Kp_second * cos + Kp_first * sin
    K_first = Kp_first * cos_vals - Kp_second * sin_vals
    K_second = Kp_second * cos_vals + Kp_first * sin_vals

    # Store K to scratch (same layout as V cache, cast back to input dtype)
    tl.store(
        k_scratch_ptr + v_first_off,
        K_first.to(V_first_raw.dtype),
        mask=slot_mask[:, None],
    )
    tl.store(
        k_scratch_ptr + v_second_off,
        K_second.to(V_first_raw.dtype),
        mask=slot_mask[:, None],
    )


def reconstruct_k_from_v(
    v_cache: torch.Tensor,
    k_scratch: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    head_size: int,
    num_kv_heads: int,
):
    """Launch reconstruction kernel: V cache → K scratch for all positions."""
    num_seqs = seq_lens.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    half_head = head_size // 2

    grid = (num_seqs * max_blocks_per_seq, num_kv_heads)

    _reconstruct_k_from_v_kernel[grid](
        v_cache_ptr=v_cache,
        k_scratch_ptr=k_scratch,
        k_norm_weight_ptr=k_norm_weight,
        cos_sin_cache_ptr=cos_sin_cache,
        block_table_ptr=block_table,
        seq_lens_ptr=seq_lens,
        block_table_stride=block_table.stride(0),
        cache_stride_blk=v_cache.stride(0),
        cache_stride_slot=v_cache.stride(1),
        cache_stride_head=v_cache.stride(2),
        cos_sin_stride=cos_sin_cache.stride(0),
        max_blocks_per_seq=max_blocks_per_seq,
        BLOCK_SIZE=block_size,
        HALF_HEAD=half_head,
    )


# --------------------------------------------------------------------------
# Backend and Impl classes
# --------------------------------------------------------------------------

class TritonAttentionKeqVBackend(TritonAttentionBackend):
    """Triton attention backend for k_eq_v layers — Gemma4 global attention.

    Cache stores only V = v_norm(k_raw); K is reconstructed from V before
    the standard attention kernel, saving 50% persistent KV cache memory.
    """

    head_size_v_cache: ClassVar[int | None] = 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_KEQV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionKeqVImpl"]:
        return TritonAttentionKeqVImpl  # type: ignore[name-defined]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # 4D layout: (num_blocks, block_size, num_kv_heads, head_size).
        # Stores V only; K is reconstructed from V at runtime.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 0, 3, 4)
        elif cache_layout == "HND":
            return (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")


class TritonAttentionKeqVImpl(TritonAttentionImpl):
    """Attention impl that stores V in cache and reconstructs K before attention.

    K = NeoX_RoPE(V * k_norm_weight) — algebraic identity from shared RMS norm.
    A lightweight Triton kernel reconstructs K for all positions into a shared
    scratch buffer, then the standard attention kernel runs unmodified.
    """

    _shared_k_scratch: ClassVar[torch.Tensor | None] = None
    _memory_reservation: ClassVar[torch.Tensor | None] = None
    _reservation_device: ClassVar[torch.device | None] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k_norm_weight: torch.Tensor | None = None
        self._rotary_emb: torch.nn.Module | None = None

    def set_kraw_params(
        self,
        k_norm_weight: torch.Tensor,
        rotary_emb: torch.nn.Module,
    ) -> None:
        """Store k_norm_weight and rotary_emb; reserve GPU memory for scratch.

        Called during model init (before KV cache profiling). The reservation
        is measured as model memory, reducing available KV cache budget.
        """
        self._k_norm_weight = k_norm_weight
        self._rotary_emb = rotary_emb

        if TritonAttentionKeqVImpl._memory_reservation is None:
            reserve_elements = (1024**3) // k_norm_weight.element_size()
            TritonAttentionKeqVImpl._memory_reservation = torch.empty(
                reserve_elements,
                dtype=k_norm_weight.dtype,
                device=k_norm_weight.device,
            )
            TritonAttentionKeqVImpl._reservation_device = k_norm_weight.device

    @classmethod
    def _get_k_scratch(cls, kv_cache: torch.Tensor) -> torch.Tensor:
        """Return (and lazily allocate) the shared K scratch buffer."""
        if (cls._shared_k_scratch is None
                or cls._shared_k_scratch.shape != kv_cache.shape
                or cls._shared_k_scratch.device != kv_cache.device):
            cls._memory_reservation = None
            cls._shared_k_scratch = torch.empty_like(kv_cache)
        return cls._shared_k_scratch

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: reconstruct K from cached V, then run standard attention."""

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not supported "
                "for TritonAttentionKeqVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            num_actual_tokens = attn_metadata.num_actual_tokens
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        # --- Reconstruct K from V for all cached positions ---
        k_scratch = self._get_k_scratch(kv_cache)

        assert self._rotary_emb is not None, (
            "set_kraw_params must be called before forward"
        )
        assert self._k_norm_weight is not None

        cos_sin_cache = self._rotary_emb.cos_sin_cache
        if cos_sin_cache.dtype != kv_cache.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=kv_cache.dtype)
        if cos_sin_cache.device != kv_cache.device:
            cos_sin_cache = cos_sin_cache.to(device=kv_cache.device)

        block_size = kv_cache.shape[1]
        num_kv_heads = kv_cache.shape[2]
        head_size = kv_cache.shape[3]

        reconstruct_k_from_v(
            v_cache=kv_cache,
            k_scratch=k_scratch,
            k_norm_weight=self._k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            block_size=block_size,
            head_size=head_size,
            num_kv_heads=num_kv_heads,
        )

        # --- Standard attention kernel (unmodified) ---
        unified_attention(
            q=query[:num_actual_tokens],
            k=k_scratch,
            v=kv_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=attn_metadata.mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=None,
            v_scale_cache=None,
            chunk_lookback=self.chunk_lookback,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Write V to the persistent cache.  K is reconstructed at attention time.

        We still write K to the scratch buffer here for the current tokens
        (it will be overwritten by the reconstruction kernel anyway, but this
        keeps the scatter write simple using the existing combined kernel).
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        k_scratch = self._get_k_scratch(kv_cache)

        triton_reshape_and_cache_flash(
            key,
            value,
            k_scratch,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
