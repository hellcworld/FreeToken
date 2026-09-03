from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from freetoken.utils import div_even, init_logger, mem_GB

logger = init_logger(__name__)


class CacheRebuildRejected(Exception):
    """A runtime cache rebuild was rejected BEFORE any destructive free (e.g. the
    requested geometry does not fit). The old caches are intact and serving continues --
    this is recoverable, unlike a failure after the free."""


# --kv-cache-dtype values that override the global dtype for the paged-KV slabs.
# "auto" (and anything else) resolves to None == follow config.dtype.
KV_CACHE_DTYPE_MAP = {
    "fp8_e4m3": torch.float8_e4m3fn,
}


def resolve_kv_dtype(config) -> torch.dtype | None:
    """The paged-KV slab dtype overriding the global one, or None to follow it.

    Pure name -> dtype mapping; the feasibility checks (backend, model family,
    device, global dtype) live in Engine._adjust_config so both the CLI and the
    programmatic EngineConfig path are covered."""
    raw = getattr(config, "kv_cache_dtype", "auto") or "auto"
    if raw == "auto":
        return None
    if raw not in KV_CACHE_DTYPE_MAP:
        raise ValueError(
            f"--kv-cache-dtype {raw!r}: unsupported (supported: "
            f"{', '.join(['auto', *KV_CACHE_DTYPE_MAP])})"
        )
    return KV_CACHE_DTYPE_MAP[raw]


def kv_elem_bytes(config) -> int:
    """Bytes per paged-KV element: the overriding kv dtype's itemsize when one is
    set (fp8 stores 1 byte/elem), else the global dtype's."""
    kv_dtype = resolve_kv_dtype(config)
    return config.dtype.itemsize if kv_dtype is None else kv_dtype.itemsize


def kv_scale_bytes_per_token_per_layer(config) -> int:
    """Per-token, per-layer bytes for the fp8-KV dynamic scales: one FP32 scale per
    (token, head) per slab, times the two slabs. 0 unless an overriding kv dtype is
    set (fp8 is the only one today; MLA latent groups store no separate scales --
    they are rejected for fp8 KV at config time)."""
    if resolve_kv_dtype(config) is None:
        return 0
    return 2 * 4


def spec_kv_bytes_per_token(spec, config) -> int:
    """One paged-KV group's bytes per token: (1|2 slabs) x head_dim x local kv heads x dtype
    x layers, plus the bf16 DSA index-key slab when the spec carries indexer dims. Pure
    per-spec arithmetic -- pool families compose it over THEIR OWN groups; no family
    branching here. (2 bytes/elem == the torch.bfloat16 dsa_pool.DSAKVCache._alloc
    hardcodes; keep the two in lockstep if the slab dtype ever changes.) An overriding
    kv-cache dtype (fp8) prices the slab at its own itemsize and adds the per-(token,
    head) FP32 scale bytes; MLA latent groups never take fp8 (rejected at config time),
    so the scale term is two-slab-only."""
    local_heads = div_even(
        spec.num_kv_heads, config.tp_info.size, allow_replicate=True
    )
    kv_bytes = kv_elem_bytes(config)
    if spec.mla:
        # fp8 KV never applies to MLA latent groups (rejected at config time and the
        # pool asserts bf16); always price the single slab at the global dtype so a
        # programmatic config that skipped the gate cannot undercount.
        kv_bytes = config.dtype.itemsize
        per_token = spec.head_dim * local_heads * kv_bytes * spec.num_layers
    else:
        per_token = 2 * spec.head_dim * local_heads * kv_bytes * spec.num_layers
        per_token += local_heads * kv_scale_bytes_per_token_per_layer(config) * spec.num_layers
    return per_token + spec.index_head_dim * spec.num_index_layers * 2


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    # Pools whose buffers are bound into per-forward model scratch (DSV4's tiers) need the
    # model re-bound after a rebuild; the engine asks before it resizes.
    needs_rebind_on_rebuild: ClassVar[bool] = False

    # ---- sizing/cost classmethods: run BEFORE the pool exists (startup budget solve,
    # --moe-cache-auto). The engine measures memory and passes bytes in; each pool family
    # implements kv_cost for ITS OWN buffers only (the engine sums families, e.g. adds
    # LinearStatePool.state_pool_bytes for hybrid-GDN models); solve_num_pages and
    # validate_rebuild are family-agnostic templates over cls.kv_cost.

    @classmethod
    @abstractmethod
    def kv_cost(cls, config, **kwargs) -> tuple[int, int, int, int]:
        """``(cache_per_page, fixed_cache_size, page_tokens, min_reserve_tokens)`` for this
        pool family's OWN buffers, priced from the group specs / model payload. No other
        pool's terms (GDN state, MoE) belong here. Extra keywords are each family's own
        convention -- the base declares none; window-bearing pools take
        ``num_swa_pages=`` (a rebuild's target window, priced before the config override
        is written)."""

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        """Largest usable page count fitting ``available_memory`` (bytes measured by the
        engine: memory_ratio x baseline free minus resident weights and any sibling pool's
        fixed cost), honoring ``config.num_page_override``."""
        cache_per_page, fixed_cache_size, _, _ = cls.kv_cost(config)
        num_pages = config.num_page_override
        if num_pages is None:
            num_pages = (available_memory - fixed_cache_size) // cache_per_page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real_kv_size = num_pages * cache_per_page + fixed_cache_size
        logger.info(
            f"Allocating {num_pages * config.page_size} tokens for KV cache, "
            f"K + V = {mem_GB(real_kv_size)}"
        )
        return num_pages

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # rebuild rejects num_pages <= 0, so the floor is one page's worth of tokens.
        return int(config.page_size)

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "", **targets,
    ) -> None:
        """Budget fit-check for a runtime rebuild target, BEFORE any destructive free.
        The engine supplies the memory account (baseline/weights, the MoE terms, and any
        sibling pool's fixed bytes at ITS target, e.g. the GDN state pool); the pool
        answers whether its own target geometry fits. Raises CacheRebuildRejected.

        The base names no family-specific target: the engine forwards the whole request
        bag through ``**targets``, overrides declare the keys they consume (DSV4:
        num_swa_pages/moe_cache_size), and this template hands each family's kv_cost
        exactly the non-None keys its signature declares -- so a new pool family adds a
        sizing target by declaring it on ITS kv_cost, with no base-signature churn."""
        import inspect

        from freetoken.engine.cache_budget import net_cache_budget_bytes, required_bytes

        target_pages = num_pages if num_pages is not None else current_num_pages
        cost_params = inspect.signature(type(self).kv_cost).parameters
        cost_kwargs = {
            k: v for k, v in targets.items() if v is not None and k in cost_params
        }
        cache_per_page, fixed_cache_size, _, _ = type(self).kv_cost(config, **cost_kwargs)
        budget = net_cache_budget_bytes(
            config.memory_ratio, baseline_free, weights_bytes,
            fixed_cache_size + extra_fixed_bytes,
        )
        need = required_bytes(target_moe, target_pages, per_expert_bytes, cache_per_page)
        if need > budget:
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, kv={target_pages} pages{extra_note}) "
                f"needs {mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    @abstractmethod
    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        """Resize to ``num_pages`` USABLE pages in place, identity-preserving (callers keep
        holding this object and re-derive views per forward). The dummy page and every
        secondary tier (window pool, index slab, state rings) are the pool's own to derive.
        Uniform-pool families (full/MLA/DSA) size from ``num_pages x page_size`` alone and
        ignore ``num_swa_pages``; window-bearing pools (SWA hybrid, DSV4) take the target
        window explicitly here, falling back to config's override/ratio when None."""

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        """Called once at engine init and again on every page-table realloc. Pools routed by
        the shared table but deriving reads through their own mapping (DSV4) re-point here."""

    @abstractmethod
    def unit_bytes(self) -> tuple[int, int]:
        """``(kv_bytes_per_token, swa_bytes_per_token)`` of the live pool -- the per-unit VRAM
        cost the cache sliders denominate in. 0 for a tier this pool does not own."""

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    def k_scale(self, index: int) -> torch.Tensor:
        """Per-(token, head) FP32 K dequant scales; only pools storing quantized KV
        (fp8 MHA today) implement this. The Triton backend reads it only when the
        pool dtype is an overriding kv dtype."""
        raise NotImplementedError(f"{type(self).__name__} stores no KV scales")

    def v_scale(self, index: int) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} stores no KV scales")

    @abstractmethod
    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # Hybrid (GDN) models: the restored GDN state snapshot slot for this prefix (None = cold /
    # non-hybrid). Surfaced by HybridRadixCache via CacheManager.match_req.
    mamba_value: int | None = None
    # TODO: support HiCache


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
