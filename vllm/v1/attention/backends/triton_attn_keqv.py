# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend for Gemma4 global attention layers (k_eq_v).

Saves ~50% KV cache memory on global layers by storing only V in the
persistent per-layer cache. K is reconstructed from V using the algebraic
identity:

    K = RoPE(V * k_norm_weight)

This works because both k_norm and v_norm are RMS norms applied to the same
k_raw input (shared weights due to k_eq_v), and v_norm has no learnable
weight. The RMS terms cancel: K_pre_rope = V * k_norm_weight.

The K reconstruction is fused into the attention kernel's tile loop (online
RoPE), similar to online softmax. This avoids a separate reconstruction
kernel and scratch buffer, saving HBM bandwidth and memory.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from vllm.config.cache import CacheDType
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    triton_reshape_and_cache_flash,
    unified_attention,
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

    # Only native-dtype cache is supported. Quantized KV cache (fp8, int8)
    # would require adding dequantization to the reconstruction kernel before
    # the V * k_norm_weight multiply — currently it reads raw V values directly.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_KEQV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionKeqVImpl"]:
        return TritonAttentionKeqVImpl

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

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # MM prefix is passed through to unified_attention and should work
        # (reconstruction kernel is agnostic to it), but untested.
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        # Sinks are passed through to unified_attention and should work
        # (reconstruction kernel processes all positions), but untested.
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class TritonAttentionKeqVImpl(TritonAttentionImpl):
    """Attention impl that stores only V in cache; K is reconstructed online.

    K = NeoX_RoPE(V * k_norm_weight) — algebraic identity from shared RMS norm.
    The reconstruction is fused into the attention kernel's tile loop (online
    RoPE), eliminating the need for a separate reconstruction kernel or scratch
    buffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k_norm_weight: torch.Tensor | None = None
        self._rotary_emb: torch.nn.Module | None = None

    def set_k_raw_params(
        self,
        k_norm_weight: torch.Tensor,
        rotary_emb: torch.nn.Module,
    ) -> None:
        """Store k_norm_weight and rotary_emb for online K reconstruction."""
        self._k_norm_weight = k_norm_weight
        self._rotary_emb = rotary_emb

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
        """Forward pass: K is reconstructed inline via online RoPE."""

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not supported "
                "for TritonAttentionKeqVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        assert self._rotary_emb is not None, (
            "set_k_raw_params must be called before forward"
        )
        assert self._k_norm_weight is not None

        cos_sin_cache = self._rotary_emb.cos_sin_cache
        if cos_sin_cache.dtype != kv_cache.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=kv_cache.dtype)
        if cos_sin_cache.device != kv_cache.device:
            cos_sin_cache = cos_sin_cache.to(device=kv_cache.device)

        # Online RoPE: pass V cache as both k and v; the kernel reconstructs
        # K inline from V using k_norm_weight + cos_sin_cache.
        unified_attention(
            q=query[:num_actual_tokens],
            k=kv_cache,
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
            k_norm_weight=self._k_norm_weight,
            cos_sin_cache=cos_sin_cache,
            rotary_pairs=self._rotary_emb.rope_angles,
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
        """Write only V to the persistent cache. K is reconstructed online."""

        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        # Only V needs to be stored; write it to both "key" and "value" slots
        # (they point to the same kv_cache tensor).
        triton_reshape_and_cache_flash(
            value,
            value,
            kv_cache,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
