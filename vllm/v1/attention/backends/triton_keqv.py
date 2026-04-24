# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend for layers where V == K (k_eq_v).

Allocates a single K slab (half the normal KV cache), and passes the same
tensor for both key_cache and value_cache. The Triton kernel writes
redundantly but harmlessly to the same cache location.
"""

from typing import ClassVar

import torch

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)


class TritonKEqVBackend(TritonAttentionBackend):
    """Triton attention backend for k_eq_v layers (K == V).

    Stores only the K slab (half normal memory).
    The same buffer is reused for both key_cache and value_cache at compute time.
    """

    kv_cache_head_size_v: ClassVar[int] = 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_KEQV"

    @staticmethod
    def get_impl_cls():
        return TritonKEqVImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # 4D layout: (num_blocks, block_size, num_kv_heads, head_size).
        # Only K is stored; V == K so same tensor is reused at compute time.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            # (num_blocks, block_size, num_kv_heads, head_size) — identity
            return (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 0, 3, 4)
        elif cache_layout == "HND":
            # (num_blocks, block_size, num_kv_heads, head_size) → swap dims 1,2
            return (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")


class TritonKEqVImpl(TritonAttentionImpl):

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
        """Forward pass for k_eq_v layers with 4D cache.

        Identical to parent forward() but doesn't unbind the cache
        (no K/V split since they're the same tensor).
        """
        from vllm.v1.attention.backends.triton_attn import unified_attention

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonKEqVImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # Check for encoder attention (no cache)
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

        # Per-token-head quantized KV cache: use separate scale caches.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache = kv_cache
            value_cache = kv_cache  # V == K
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = key_cache
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._k_scale_cache  # K == V
        # FP8 per-tensor / auto path (original flow).
        else:
            key_cache = kv_cache
            value_cache = kv_cache  # V == K
            if is_quantized_kv_cache(self.kv_cache_dtype):
                if key_cache.dtype != self.fp8_dtype:
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = key_cache
            k_descale = None  # For k_eq_v, K == V so no separate scales needed
            v_descale = None
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        # kv_cache: (num_blocks, block_size, num_kv_heads, head_size) — K only.
        # For k_eq_v, pass the same cache for both K and V.
        # The kernel will write key/value redundantly (both to same location),
        # which is harmless and avoids the memory overhead of storing V separately.

        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache = kv_cache
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
            value_cache = key_cache  # V == K
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                key,  # V == K
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._k_scale_cache,  # no separate v_scale
                slot_mapping,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        key_cache = kv_cache
        value_cache = kv_cache  # V == K
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = key_cache
        triton_reshape_and_cache_flash(
            key,
            key,  # V == K
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._k_scale,  # no separate v_scale
        )
