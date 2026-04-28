# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend for Gemma4 global attention layers (k_eq_v).

Stores V = v_norm(k_raw) in a single 4D cache slab (no K/V split), saving
50% KV cache memory vs. the standard 5D layout.  K is reconstructed inside
the attention kernel via:

    K[d] = V[d] * k_norm_weight[d] * cos(pos, d)
           + sign(d) * V[d_partner] * k_norm_weight[d_partner] * sin(pos, d)

where d_partner = d ± HEAD_SIZE/2  (neox-style rotation),
      cos/sin come from the rotary embedding's cos_sin_cache,
      and sign = -1 for d < HEAD_SIZE/2, +1 otherwise.

For Gemma4's proportional RoPE (partial_rotary_factor=0.25) the
non-rotated dims already have cos=1, sin=0 in the cache, so the formula
degenerates to identity for those dims automatically.
"""

from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    triton_reshape_and_cache_flash,
)


class TritonAttentionKeqVBackend(TritonAttentionBackend):
    """Triton attention backend for k_eq_v layers — Gemma4 global attention.

    Cache stores only V = v_norm(k_raw); K is reconstructed inline in the
    attention kernel using k_norm weights and RoPE tables, saving 50% memory.
    """

    head_size_v_cache: ClassVar[int | None] = 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_KEQV"

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
        # Stores V only; K is reconstructed at attention time.
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
    """Attention impl that stores V in the cache and reconstructs K on read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set by set_kraw_params() after the layer is built in the model.
        self._k_norm_weight: torch.Tensor | None = None
        self._cos_sin_cache: torch.Tensor | None = None

    def set_kraw_params(
        self,
        k_norm_weight: torch.Tensor,
        rotary_emb: torch.nn.Module,
    ) -> None:
        """Register k_norm weight and RoPE tables for K reconstruction.

        Called once from Gemma4Attention.__init__ after the Attention layer is
        created.

        Args:
            k_norm_weight: RMSNorm weight of shape (head_size,).
            rotary_emb: The layer's RotaryEmbedding module; must expose
                        ``cos_sin_cache`` of shape (max_positions, head_size)
                        with cos in [:head_size//2] and sin in [head_size//2:].
        """
        self._k_norm_weight = k_norm_weight
        if not hasattr(rotary_emb, "cos_sin_cache"):
            raise AttributeError(
                "rotary_emb has no cos_sin_cache; cannot use TritonAttentionKeqVImpl"
            )
        self._cos_sin_cache = rotary_emb.cos_sin_cache

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
        """Forward pass for k_eq_v layers.

        kv_cache stores V = v_norm(k_raw).  The attention kernel reconstructs
        K inline using k_norm_weight and cos_sin_cache.
        """
        from vllm.v1.attention.backends.triton_attn import unified_attention

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

        if self._k_norm_weight is None or self._cos_sin_cache is None:
            raise RuntimeError(
                "TritonAttentionKeqVImpl.set_kraw_params() was never called. "
                "Call it from Gemma4Attention.__init__ after creating Attention."
            )

        num_actual_tokens = attn_metadata.num_actual_tokens

        # 4D cache stores V = v_norm(k_raw); no K/V split.
        # Quantized path not yet supported for this backend.
        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        v_cache = kv_cache  # stores V; K is reconstructed by the kernel

        # Ensure cos_sin_cache is on the same device as the query.
        cos_sin = self._cos_sin_cache
        if cos_sin.device != query.device:
            cos_sin = cos_sin.to(query.device)

        k_norm_w = self._k_norm_weight
        if k_norm_w.device != query.device:
            k_norm_w = k_norm_w.to(query.device)

        unified_attention(
            q=query[:num_actual_tokens],
            k=v_cache,   # key_cache_ptr reads V; kernel reconstructs K
            v=v_cache,   # value_cache_ptr reads V directly
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
            # V-cache K reconstruction:
            k_norm_weight=k_norm_w,
            cos_sin_cache=cos_sin,
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
        """Write V = v_norm(k_raw) to the 4D cache.

        ``value`` is v_norm(k_raw) — already computed by Gemma4Attention.forward.
        ``key`` (processed K = k_norm + RoPE) is not stored; it is discarded.
        K is reconstructed from the cached V at attention time.
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        if self._is_per_token_head_quant or is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Quantized KV cache is not yet supported for TritonAttentionKeqVImpl"
            )

        # Write V to the single 4D cache slab.
        # key_cache == value_cache (same 4D tensor), so the kernel writes V
        # to both K and V slots which happen to be the same memory location.
        triton_reshape_and_cache_flash(
            value,      # write V, not key
            value,      # same data for both cache halves (they share storage)
            kv_cache,   # key_cache  = 4D slab
            kv_cache,   # value_cache = same 4D slab
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
