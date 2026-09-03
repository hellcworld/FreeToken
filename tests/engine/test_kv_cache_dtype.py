"""--kv-cache-dtype fp8_e4m3: config-time validation and auto-backend steering.

fp8 paged KV dequantizes inside the Triton attention kernels only, so the matrix
here pins: auto must collapse the FULL tree to triton, every explicit vendor
backend must be rejected, and every pool family / global dtype / device that
cannot host the dequant must fail loudly at config time (before the pool
allocates). Pure config-level checks -- no pool, no kernel, no CUDA required.
"""

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.models.config import KVCacheGroupSpec


def _spec(name, attn_type, *, mla=False, index_head_dim=0):
    return KVCacheGroupSpec(
        name=name,
        layer_ids=(0, 1),
        num_kv_heads=1,
        head_dim=64,
        sliding_window=None,
        mla=mla,
        index_head_dim=index_head_dim,
        num_index_layers=2 if index_head_dim else 0,
        attn_type=attn_type,
    )


def _model_config(kind):
    from types import SimpleNamespace

    mc = SimpleNamespace(
        model_type=kind,
        single_stream_only=False,
        is_moe=False,
        expert_quant="none",
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=4,
        rotary_config=SimpleNamespace(max_position=1024),
    )
    if kind == "full":
        specs = (_spec("full", AttnType.FULL),)
    elif kind == "swa":
        mc.has_swa_attention = True
        specs = (
            _spec("full", AttnType.FULL),
            _spec("swa", AttnType.SWA),
        )
    elif kind == "mla":
        specs = (_spec("full", AttnType.MLA, mla=True),)
    elif kind == "dsa":
        specs = (_spec("full", AttnType.DSA, mla=True, index_head_dim=128),)
    elif kind == "dsv4":
        mc.dsv4_args = SimpleNamespace(window_size=128)
        specs = (_spec("dsv4", AttnType.DSV4),)
    elif kind == "bsa":
        specs = (_spec("full", AttnType.BSA, index_head_dim=128),)
    else:
        raise AssertionError(kind)
    mc.kv_cache_group_specs = lambda: specs
    return mc


def _config(kind="full", **overrides):
    from types import SimpleNamespace

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        **overrides,
    )
    object.__setattr__(config, "model_config", _model_config(kind))
    return config


def _patch_env(monkeypatch, *, cuda=True, flashinfer=True, sgl=True, major=9):
    from freetoken.engine import engine

    # The fp8 gate includes a real device check; the config matrix itself must be
    # runnable on CUDA-less machines, so pin it like the package/arch gates.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(engine, "is_sm100_family", lambda: major == 10)
    monkeypatch.setattr(engine, "is_sm90_family", lambda: major == 9)
    monkeypatch.setattr(engine, "_flashinfer_available", lambda: flashinfer)
    monkeypatch.setattr(engine, "_sgl_flash_attn_available", lambda: sgl)


def test_auto_resolves_triton_despite_vendor_availability(monkeypatch):
    # On sm90 with flashinfer + sgl_kernel installed, a full-attention model
    # resolves to "fa,fi" -- unless fp8 KV asks for in-kernel dequant, which only
    # triton hosts. The fp8 tree must collapse to triton on every arch.
    from freetoken.engine.engine import _adjust_config

    for major in (9, 10):
        _patch_env(monkeypatch, major=major)
        config = _config("full", attention_backend="auto", kv_cache_dtype="fp8_e4m3")
        _adjust_config(config)
        assert config.attention_backend == "triton"


def test_fp8_accepts_explicit_triton(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    _adjust_config(config)
    assert config.attention_backend == "triton"


@pytest.mark.parametrize("backend", ["fa", "fi", "trtllm", "fa,fi"])
def test_fp8_rejects_vendor_backends(monkeypatch, backend):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, major=10)  # arch/package gates satisfied
    config = _config("full", attention_backend=backend, kv_cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="Triton attention kernels"):
        _adjust_config(config)


def test_fp8_requires_bfloat16_global_dtype(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    object.__setattr__(config, "dtype", torch.float16)
    with pytest.raises(ValueError, match="bfloat16"):
        _adjust_config(config)


def test_fp8_requires_cuda(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch, cuda=False)
    config = _config("full", attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="CUDA"):
        _adjust_config(config)


@pytest.mark.parametrize("kind", ["dsv4", "bsa"])
def test_fp8_rejects_dsv4_and_bsa_pools(monkeypatch, kind):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config(kind, attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="DSV4/BSA"):
        _adjust_config(config)


@pytest.mark.parametrize("kind", ["mla", "dsa"])
def test_fp8_rejects_mla_and_dsa_latents(monkeypatch, kind):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config(kind, attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="MLA/DSA"):
        _adjust_config(config)


def test_fp8_rejects_swa_window_pool(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("swa", attention_backend="triton", kv_cache_dtype="fp8_e4m3")
    with pytest.raises(ValueError, match="SWA"):
        _adjust_config(config)


def test_unknown_kv_cache_dtype_name_rejected(monkeypatch):
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="triton", kv_cache_dtype="int4")
    with pytest.raises(ValueError, match="unsupported"):
        _adjust_config(config)


def test_auto_kv_cache_dtype_keeps_historical_backend_tree(monkeypatch):
    # Without the fp8 flag the resolution tree is unchanged: sm90 + vendors -> fa,fi.
    from freetoken.engine.engine import _adjust_config

    _patch_env(monkeypatch)
    config = _config("full", attention_backend="auto", kv_cache_dtype="auto")
    _adjust_config(config)
    assert config.attention_backend == "fa,fi"
