# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for layers where V == K (k_eq_v).

Allocates a single K slab (half the normal KV cache), writes only K, and
passes the same tensor for key_cache and value_cache in the forward pass.
"""

from typing import ClassVar

import torch

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    cascade_attention,
)


class FlashAttentionKEqVBackend(FlashAttentionBackend):
    """KV cache backend for attention layers where V == K.

    Stores only the K slab (half normal memory).
    The same buffer is reused for both key_cache and value_cache at compute time.
    """

    # Override: cache only needs head_size bytes per head (K only, not K+V).
    # self.head_size_v in Attention stays at head_size for correct output sizing.
    kv_cache_head_size_v: ClassVar[int] = 0

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_KEQV"

    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return FlashAttentionKEqVImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # 4D layout: (num_blocks, block_size, num_kv_heads, head_size).
        # Only K is stored; V == K so the same tensor is reused at compute time.
        # Bytes per block = block_size * num_kv_heads * head_size * dtype
        # = exactly half of the standard (2, num_blocks, ...) layout.
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


class FlashAttentionKEqVImpl(FlashAttentionImpl):

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        # kv_cache: (num_blocks, block_size, num_kv_heads, head_size) — K only.
        # Pass key for both K and V args; single slab serves as both caches.
        reshape_and_cache_flash(
            key,
            key,       # V == K
            kv_cache,
            kv_cache,  # single slab for both K and V
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._k_scale,  # no separate v_scale; K scale applies to V too
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale=None,
        output_block_scale=None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization not yet supported for "
                "FlashAttentionKEqVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # kv_cache: (num_blocks, block_size, num_kv_heads, head_size).
        # K == V: reuse the single slab for both cache tensors.
        key_cache = kv_cache
        value_cache = kv_cache  # V equals K for this layer

        if is_quantized_kv_cache(self.kv_cache_dtype):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = key_cache  # keep alias after view

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata
            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
            sliding_window_size = (
                list(self.sliding_window)
                if self.sliding_window is not None
                else None
            )

            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape),
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._k_scale.expand(descale_shape),  # V == K
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._k_scale,  # V == K
            s_aux=self.sinks,
        )
        return output
