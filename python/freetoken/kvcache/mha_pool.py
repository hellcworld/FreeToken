from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool

# e4m3's largest finite magnitude; the amax scale maps every row's |max| onto it.
E4M3_MAX = 448.0


def quantize_rows_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(tokens, kv_heads, head_dim)`` rows onto the e4m3 grid with one
    dynamic FP32 amax scale per (token, head) row: the row's |max| maps to ``E4M3_MAX``
    so every element sits in the format's sweet spot (per-tensor scales would waste
    range on row-magnitude spread). Returns ``(fp8 rows, fp32 scales)``; the cast is
    round-to-nearest-even, so exactly-representable values pass through unchanged and
    the worst-case error is the e4m3 half-ulp, 2^-4 relative (bf16: 2^-9)."""
    amax = x.abs().amax(dim=-1).float()
    # all-zero rows (padding) would divide by zero -> NaN; a tiny scale keeps them 0.
    scale = (amax / E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = (x.float() / scale.unsqueeze(-1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def dequantize_rows_e4m3(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The exact inverse of :func:`quantize_rows_e4m3` (up to the stored quantization
    itself): fp8 -> fp32 is exact and the scale multiply is a single rounding."""
    return q.to(torch.float32) * scale.unsqueeze(-1)


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        # fp8 paged KV (--kv-cache-dtype fp8_e4m3): slabs hold e4m3 values, a parallel
        # FP32 slab holds the per-(token, head) scales. Every other dtype stores plain
        # 16-bit rows with no scale bookkeeping.
        self._is_fp8 = dtype == torch.float8_e4m3fn
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._device = device
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads, head_dim, dtype)
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def _alloc(
        self,
        num_storage_layers: int,
        num_pages: int,
        page_size: int,
        local_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=self._device,
            dtype=dtype,
        )
        if self._is_fp8:
            self._scale_buffer: torch.Tensor | None = torch.empty(
                (2, num_storage_layers, num_pages * page_size, local_kv_heads),
                device=self._device,
                dtype=torch.float32,
            )
        else:
            self._scale_buffer = None
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        self._k_buffer = None
        self._v_buffer = None
        self._scale_buffer = None
        self._kv_buffer = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._alloc(num_storage_layers, num_pages, page_size, local_kv_heads, head_dim, dtype)
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        # MTP draft head: its attention layers each hold one more full-attention slab.
        # spec_kv_bytes_per_token prices a WHOLE group (x spec.num_layers), so derive the
        # per-layer cost before charging the draft slabs -- charging the group total
        # per draft layer overprices KV by the full group's layer count (16x on a
        # 16-of-64-full hybrid like Qwen3.8-27B).
        mtp_ids = getattr(config.model_config, "mtp_layer_ids", ())
        if mtp_ids:
            full = next(
                spec
                for spec in config.model_config.kv_cache_group_specs()
                if not spec.is_swa
            )
            per_layer = spec_kv_bytes_per_token(full, config) // max(full.num_layers, 1)
            per_token += len(mtp_ids) * per_layer
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        bytes_per_token = int(buf.numel() * buf.element_size()) // tokens
        if self._scale_buffer is not None:
            bytes_per_token += self._scale_buffer.numel() * self._scale_buffer.element_size() // tokens
        return bytes_per_token, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scale(self, index: int) -> torch.Tensor:
        """Per-(token, head) FP32 K scales, fp8 KV only; (num_tokens, kv_heads)."""
        assert self._scale_buffer is not None, "scales exist only for fp8 paged KV"
        return self._scale_buffer[0, self._dense(index)]

    def v_scale(self, index: int) -> torch.Tensor:
        """Per-(token, head) FP32 V scales, fp8 KV only; (num_tokens, kv_heads)."""
        assert self._scale_buffer is not None, "scales exist only for fp8 paged KV"
        return self._scale_buffer[1, self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        k_rows = k
        v_rows = v
        if self._is_fp8:
            # k/v arrive flat (tokens, kv_heads*head_dim); quantize per (token, head)
            # row and hand store_cache the same 2D layout it gets for 16-bit stores.
            head_dim = self._storage_shape[-1]
            k_q, k_scales = quantize_rows_e4m3(k.reshape(k.shape[0], -1, head_dim))
            v_q, v_scales = quantize_rows_e4m3(v.reshape(v.shape[0], -1, head_dim))
            # torch index_copy_ takes int64 only (the CUDA store kernel takes either)
            loc = out_loc.to(torch.long)
            self._scale_buffer[0, dense].index_copy_(0, loc, k_scales)
            self._scale_buffer[1, dense].index_copy_(0, loc, v_scales)
            k_rows = k_q.view(k_q.shape[0], -1)
            v_rows = v_q.view(v_q.shape[0], -1)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k_rows,
            v=v_rows,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
