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


class TritonKEqVImpl(TritonAttentionImpl):

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
