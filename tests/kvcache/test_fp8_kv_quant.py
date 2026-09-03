"""fp8 paged KV: quantization round-trip precision and budget accounting.

CPU-only checks of the pieces that need no CUDA: the store-side quantizer
(RNE e4m3 cast under a per-(token, head) dynamic amax scale), its fp32 dequant
inverse, the pool's slab geometry for an fp8 dtype, and the spec-level byte
accounting the startup budget and runtime rebuilds price the pool with. The
CUDA-side store/dequant-kernel numerics live in tests/kernels/test_fp8_kv_attention.py.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.mha_pool import dequantize_rows_e4m3, quantize_rows_e4m3
from freetoken.models.config import KVCacheGroupSpec

# e4m3's largest finite magnitude and half-ulp relative error (4-bit exponent,
# 3-bit mantissa; bf16's 8-bit mantissa gives 2**-9).
E4M3_MAX = 448.0
E4M3_HALF_ULP_REL = 2.0**-4


def _spec(num_kv_heads=4, head_dim=128, num_layers=2):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=tuple(range(num_layers)),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        attn_type=AttnType.FULL,
    )


def _config(kv_cache_dtype="auto"):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        kv_cache_dtype=kv_cache_dtype,
    )


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def test_exact_values_round_trip_bit_exact():
    x = torch.tensor(
        [0.0, -0.0, 0.5, -0.5, 1.0, 2.0, 448.0, -448.0, 288.0],
        dtype=torch.bfloat16,
    ).view(1, 1, -1)
    q, scale = quantize_rows_e4m3(x)
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    restored = dequantize_rows_e4m3(q, scale)
    assert torch.equal(restored, x.float())


def test_random_rows_stay_within_the_format_bound():
    torch.manual_seed(0)
    for scale_spread in (1.0, 100.0, 0.01):
        x = torch.randn(32, 4, 128, dtype=torch.bfloat16) * scale_spread
        q, s = quantize_rows_e4m3(x)
        restored = dequantize_rows_e4m3(q, s)
        err = (restored - x.float()).abs()
        row_amax = x.float().abs().amax(dim=-1, keepdim=True)
        # absolute error is bounded by the largest e4m3 half-ulp of the row
        # (top-of-range step 16 in units of the amax scale == amax/28); this also
        # covers subnormal-flushed near-zero elements, whose relative error is
        # unbounded by construction.
        assert err.max().item() <= (row_amax * (16.0 / E4M3_MAX)).max().item() * 1.001
        # away from zero the relative error respects the half-ulp directly
        rel = err / x.float().abs().clamp_min(row_amax * 0.01)
        assert rel.max().item() <= E4M3_HALF_ULP_REL * 1.001


def test_zero_rows_produce_zero_not_nan():
    x = torch.zeros(3, 2, 16, dtype=torch.bfloat16)
    q, s = quantize_rows_e4m3(x)
    assert (s > 0).all()
    restored = dequantize_rows_e4m3(q, s)
    assert torch.isfinite(restored).all() and (restored == 0).all()


def test_per_row_scales_fit_each_rows_range():
    # two rows of very different magnitude must get independent scales, so both
    # quantize with the format's full relative precision (a shared scale would
    # crush the small row).
    x = torch.zeros(2, 1, 64, dtype=torch.bfloat16)
    x[0, 0, 0] = 1.0
    x[1, 0, 0] = 1e-4
    q, s = quantize_rows_e4m3(x)
    assert s[0, 0] > s[1, 0] * 1000
    restored = dequantize_rows_e4m3(q, s)
    assert restored[1, 0, 0].item() == pytest.approx(1e-4, rel=E4M3_HALF_ULP_REL * 2)


def test_fp8_pool_allocates_value_and_scale_slabs():
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=8,
        num_pages=4,
        page_size=2,
        dtype=torch.float8_e4m3fn,
        device=torch.device("cpu"),
    )
    assert pool._kv_buffer.shape == (2, 3, 4, 2, 2, 8)
    assert pool._kv_buffer.dtype == torch.float8_e4m3fn
    assert pool._scale_buffer.shape == (2, 3, 8, 2)
    assert pool._scale_buffer.dtype == torch.float32
    assert pool.k_cache(0).dtype == torch.float8_e4m3fn
    assert pool.k_scale(0).shape == (8, 2) and pool.v_scale(1).shape == (8, 2)
    # unit cost spans every storage layer: value bytes (1 B/elem: 2 slabs x
    # 3 layers x 2 heads x 8) + scale bytes (fp32 per (token, head) per slab:
    # 2 slabs x 3 layers x 2 heads x 4 B)
    value_bytes, _ = pool.unit_bytes()
    assert value_bytes == 2 * 3 * 2 * 8 * 1 + 2 * 3 * 2 * 4


def test_bf16_pool_has_no_scale_slab():
    from freetoken.kvcache.mha_pool import MHAKVCache

    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=8,
        num_pages=4,
        page_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert pool._scale_buffer is None
    value_bytes, _ = pool.unit_bytes()
    assert value_bytes == 2 * 3 * 2 * 8 * 2


def test_spec_bytes_price_fp8_slab_plus_scales():
    cfg16, cfg8 = _config("auto"), _config("fp8_e4m3")
    spec = _spec(num_kv_heads=4, head_dim=128, num_layers=2)
    bf16 = 2 * 128 * 4 * 2 * 2
    assert spec_kv_bytes_per_token(spec, cfg16) == bf16
    # slabs halve; scales add one fp32 per (token, head) per slab per layer
    fp8 = 2 * 128 * 4 * 1 * 2 + 4 * 2 * 4 * 2
    assert spec_kv_bytes_per_token(spec, cfg8) == fp8
    assert fp8 / bf16 == pytest.approx(0.5156, abs=1e-3)  # ~1.94x capacity


def test_spec_bytes_keep_mla_at_global_dtype_even_under_fp8():
    # fp8 KV never applies to MLA latent groups; a programmatic config that
    # skipped the engine gate must still not undercount the bf16 latent slab.
    from freetoken.kvcache.base import spec_kv_bytes_per_token as bytes_per_token

    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=(0, 1),
        num_kv_heads=1,
        head_dim=576,
        sliding_window=None,
        mla=True,
        attn_type=AttnType.MLA,
    )
    assert bytes_per_token(spec, _config("fp8_e4m3")) == 576 * 1 * 2 * 2
