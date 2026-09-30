"""qwen3_5_moe quantization detection + dense-pass loading for the Qwen3.8-27B dialects.

Covers the three ways a Qwen3.x-27B NVFP4 checkpoint can present itself:

* modelopt MIXED_PRECISION (nvidia/RadixArk style) -- via config.json's
  ``quantized_layers`` and, when an export strips that copy, via the
  ``hf_quant_config.json`` fallback (the #139 misroute: without the fallback the
  checkpoint parses as unquantized and lands in the pure-NVFP4 loop, which
  dequantizes every tensor through GPU kernel launches at load);
* compressed-tensors NVFP4 (llm-compressor pure export) -- unchanged behavior;
* compressed-tensors MIXED precision (e.g. unsloth/Qwen3.8-27B-NVFP4) -- an 8-bit
  float group alongside the NVFP4 one: FP8 per-channel attention/GDN/lm_head
  projections plus FP8-quantized trailing dense-MLP layers, which the dense pass
  folds into the native NVFP4 buffers via a load-time RTN requant.

The end-to-end cases drive ``iter_weights`` off tiny synthetic checkpoints written
with safetensors (a handful of MiB), asserting the emitted (name, shape, dtype)
inventory against the model's buffer contract -- the same contract verified
against the real 27B checkpoints during development.
"""

from __future__ import annotations

import json
import os

import pytest
import safetensors.torch
import torch

H = 64          # hidden
I = 128         # mlp intermediate
V = 256         # vocab
Q_ROWS = 64     # self_attn q/o rows for the synthetic layers
GDN_ROWS = 96    # linear_attn in_proj_qkv/z/out rows


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) and k == "text_config" else v)


def _text_config(num_layers: int = 2) -> dict:
    layer_types = ["linear_attention", "full_attention"][:num_layers] * 2
    return {
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": V,
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "num_experts": 0,  # dense variant (Qwen3.6/3.8-27B)
        "layer_types": layer_types[:num_layers],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    }


def _hf_config(quantization_config: dict | None = None) -> _Cfg:
    data: dict = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": _text_config(),
    }
    if quantization_config is not None:
        data["quantization_config"] = quantization_config
    return _Cfg(data)


_MODELOPT_LAYERS = {
    f"model.language_model.layers.{i}.{proj}": {"quant_algo": algo}
    for i in range(2)
    for proj, algo in (
        ("self_attn.q_proj", "FP8"), ("self_attn.k_proj", "FP8"), ("self_attn.v_proj", "FP8"),
        ("self_attn.o_proj", "FP8"), ("mlp.gate_proj", "W4A16_NVFP4"),
        ("mlp.up_proj", "W4A16_NVFP4"), ("mlp.down_proj", "W4A16_NVFP4"),
    )
    if (i == 1 or "self_attn" not in proj)
} | {"lm_head": {"quant_algo": "W4A16_NVFP4"}}

_CV_NVFP4_WEIGHTS = {
    "num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group",
}
_CT_PURE_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {"group_0": {"weights": _CV_NVFP4_WEIGHTS, "targets": ["Linear"]}},
}
_CT_MIXED_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "group_0": {"weights": {"num_bits": 8, "type": "float", "strategy": "channel"}},
        "group_1": {"weights": _CV_NVFP4_WEIGHTS},
    },
    "ignore": ["vision_tower.*", "mtp.*", "language_model.model.linear_attn.*"],
}


# --------------------------------------------------------------------- detection
def test_modelopt_mixed_detection_via_config_json():
    from freetoken.models.qwen3_5_moe.config import parse_config

    cfg = parse_config(_hf_config({
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": _MODELOPT_LAYERS,
    }))
    assert cfg.attn_quant == "fp8_pertensor"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "nvfp4"
    assert not cfg.moe_enabled


def test_modelopt_detection_falls_back_to_hf_quant_config(tmp_path):
    """Exports that strip quantization_config from config.json (Desktop catalog builds)
    must still detect via hf_quant_config.json instead of parsing as unquantized (#139)."""
    from freetoken.models.qwen3_5_moe.config import parse_config
    from freetoken.utils import cached_load_hf_config

    (tmp_path / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": _text_config(),
        }), encoding="utf-8",
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"producer": "modelopt", "quantization": {
            "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
            "quantized_layers": _MODELOPT_LAYERS,
        }}), encoding="utf-8",
    )
    cfg = parse_config(cached_load_hf_config(str(tmp_path)))
    assert cfg.attn_quant == "fp8_pertensor"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "nvfp4"


def test_ct_pure_nvfp4_detection_unchanged():
    from freetoken.models.qwen3_5_moe.config import parse_config

    cfg = parse_config(_hf_config(_CT_PURE_QUANT))
    assert cfg.attn_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "none"


def test_ct_mixed_fp8_detection():
    from freetoken.models.qwen3_5_moe.config import parse_config

    cfg = parse_config(_hf_config(_CT_MIXED_QUANT))
    assert cfg.attn_quant == "fp8_pertensor"  # FP8 per-channel == the per-row W8A16 scales
    assert cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "none"        # fp8 lm_head dequantizes to bf16


def test_ct_mxfp4_geometry_still_rejected():
    from freetoken.models.config import detect_compressed_tensors_nvfp4

    with pytest.raises(ValueError, match="unsupported compressed-tensors"):
        detect_compressed_tensors_nvfp4(_hf_config({
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {
                "num_bits": 4, "type": "float", "group_size": 32, "strategy": "tensor_group",
            }}},
        }))


# ------------------------------------------------------- ct expert-bank keys
def test_ct_expert_key_pattern_accepts_both_canonical_prefixes():
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_CT_EXPERT_KEY_RE

    for prefix in ("language_model.model", "model.language_model"):
        m = _NVFP4_CT_EXPERT_KEY_RE.match(
            f"{prefix}.layers.3.mlp.experts.17.up_proj.weight_packed"
        )
        assert m is not None, prefix
        assert m.group("layer") == "3" and m.group("expert") == "17"
        assert m.group("proj") == "up_proj" and m.group("kind") == "weight_packed"
    assert _NVFP4_CT_EXPERT_KEY_RE.match(
        "model.language_model.layers.3.mlp.experts.17.down_proj.weight_global_scale"
    ).group("kind") == "weight_global_scale"
    # mtp head experts and the un-wrapped text prefix must never match
    assert _NVFP4_CT_EXPERT_KEY_RE.match(
        "mtp.layers.0.mlp.experts.1.gate_proj.weight_packed"
    ) is None
    assert _NVFP4_CT_EXPERT_KEY_RE.match(
        "model.layers.0.mlp.experts.1.gate_proj.weight_packed"
    ) is None
    # modelopt suffixes belong to the other spec, not the ct one
    assert _NVFP4_CT_EXPERT_KEY_RE.match(
        "model.language_model.layers.0.mlp.experts.1.gate_proj.weight_scale_2"
    ) is None


# ------------------------------------------------------------------ fusion guard
def test_try_fuse_rejects_inconsistent_parts():
    from freetoken.models.qwen3_5_moe.weight import _try_fuse

    buf: dict = {}
    assert _try_fuse("model.layers.1.self_attn.q_proj.weight", torch.zeros(8, 4, dtype=torch.bfloat16), buf) == ()
    assert _try_fuse("model.layers.1.self_attn.k_proj.weight", torch.zeros(8, 4, dtype=torch.float8_e4m3fn), buf) == ()
    with pytest.raises(ValueError, match="qkv_proj.*dtype"):
        _try_fuse(  # group completes; k's dtype differs from q's -> refuse the cat
            "model.layers.1.self_attn.v_proj.weight",
            torch.zeros(8, 4, dtype=torch.float8_e4m3fn), buf,
        )


# ------------------------------------------------------------- NVFP4 RTN math
def test_nvfp4_rtn_roundtrip_within_e2m1_bound():
    from freetoken.models.nvfp4_quant import dequant_nvfp4_reference, quantize_nvfp4_rtn

    torch.manual_seed(11)
    w = torch.randn(96, 128, dtype=torch.bfloat16) * 0.05
    packed, scale, global_scale = quantize_nvfp4_rtn(w)
    assert packed.shape == (96, 64) and packed.dtype == torch.uint8
    assert scale.shape == (96, 8) and scale.dtype == torch.float8_e4m3fn
    assert global_scale.dtype == torch.float16
    deq = dequant_nvfp4_reference(packed, scale, global_scale, dtype=torch.float32)
    bound = w.float().abs().max() / 6 * 1.02  # e2m1's widest half-gap is 1.0 fp4 units
    assert (deq - w.float()).abs().max() < bound


# ------------------------------------------------- synthetic mixed-CT checkpoint
def _packed_nvfp4(rows: int, cols: int = H, seed: int = 0):
    """(weight_packed, weight_scale, weight_global_scale) tensors, CT storage."""
    from freetoken.models.nvfp4_quant import quantize_nvfp4_rtn

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float32) * 0.05
    packed, scale, global_scale = quantize_nvfp4_rtn(w)
    return w, packed, scale, float(global_scale)


def _write_ckpt(path, config: dict, tensors: dict[tuple[str, torch.Tensor]]):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    safetensors.torch.save_file(
        tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"}
    )
    return str(path)


def _mixed_ct_tensors(mlp_fp8_layers: set[int]):
    """A 2-layer synthetic unsloth-style checkpoint: layer 0 linear (GDN), layer 1 full
    attention. MLP layers in ``mlp_fp8_layers`` are stored per-channel FP8 (folded to
    NVFP4 at load); the rest are stored packed."""
    t: dict[tuple[str, torch.Tensor]] = {}
    t["model.language_model.embed_tokens.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    t["model.language_model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)

    fp8 = torch.float8_e4m3fn
    def _fp8(rows, cols, key):
        w = torch.randn(rows, cols, dtype=torch.float32).clamp(-2, 2) * 0.1
        t[f"{key}.weight"] = w.to(fp8)
        t[f"{key}.weight_scale"] = torch.rand(rows, 1, dtype=torch.float32) + 0.5
        return w

    # layer 0: linear attention
    p = "model.language_model.layers.0"
    for proj in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"):
        _fp8(GDN_ROWS, H, f"{p}.{proj}")
    for proj, rows in (("linear_attn.in_proj_b", 16), ("linear_attn.in_proj_a", 16)):
        t[f"{p}.{proj}.weight"] = torch.randn(rows, H, dtype=torch.bfloat16) * 0.05
    t[f"{p}.linear_attn.conv1d.weight"] = torch.randn(32, 1, 4, dtype=torch.bfloat16) * 0.05
    t[f"{p}.linear_attn.A_log"] = torch.randn(4, dtype=torch.float32) * 0.1
    t[f"{p}.linear_attn.dt_bias"] = torch.randn(4, dtype=torch.float32) * 0.1
    t[f"{p}.linear_attn.norm.weight"] = torch.ones(16, dtype=torch.bfloat16)

    # layer 1: full attention
    p1 = "model.language_model.layers.1"
    for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
        _fp8(Q_ROWS, H, f"{p1}.{proj}")
    t[f"{p1}.self_attn.q_norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    t[f"{p1}.self_attn.k_norm.weight"] = torch.ones(8, dtype=torch.bfloat16)

    for li, pfx in ((0, p), (1, p1)):
        for n in ("input_layernorm", "post_attention_layernorm"):
            t[f"{pfx}.{n}.weight"] = torch.ones(H, dtype=torch.bfloat16)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            rows, cols = (I, H) if proj != "down_proj" else (H, I)
            key = f"{pfx}.mlp.{proj}"
            if li in mlp_fp8_layers:
                _fp8(rows, cols, key)
            else:
                _, packed, scale, wg = _packed_nvfp4(rows, cols, seed=li)
                t[f"{key}.weight_packed"] = packed
                t[f"{key}.weight_scale"] = scale
                t[f"{key}.weight_global_scale"] = torch.tensor([wg], dtype=torch.float32)
                t[f"{key}.input_global_scale"] = torch.tensor([1.0], dtype=torch.float32)

    lw = torch.randn(V, H, dtype=torch.float32) * 0.02
    t["lm_head.weight"] = lw.to(fp8)
    t["lm_head.weight_scale"] = torch.rand(V, 1, dtype=torch.float32) + 0.5
    return t


def _iter(path):
    from freetoken.models.qwen3_5_moe.weight import iter_weights

    return dict(
        (name, tensor)
        for name, tensor in iter_weights(
            str(path), torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    )


def test_mixed_ct_checkpoint_emission_contract(tmp_path):
    out = _iter(_write_ckpt(
        tmp_path, {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": _text_config(),
            "quantization_config": _CT_MIXED_QUANT,
        },
        _mixed_ct_tensors(mlp_fp8_layers={1}),  # layer-1 MLP arrives FP8 -> RTN fold
    ))
    fp8 = torch.float8_e4m3fn

    # full-attn layer: fused fp8 qkv (per-row scales), standalone fp8 o_proj
    qkv = out["model.layers.1.self_attn.qkv_proj.weight"]
    assert qkv.dtype == fp8 and qkv.shape[0] == 3 * Q_ROWS  # q|k|v fused parts
    assert out["model.layers.1.self_attn.qkv_proj.weight_scale"].dtype == torch.float32
    assert out["model.layers.1.self_attn.qkv_proj.weight_scale"].shape == (3 * Q_ROWS,)
    assert out["model.layers.1.self_attn.o_proj.weight"].dtype == fp8
    assert out["model.layers.1.self_attn.o_proj.weight_scale"].shape == (Q_ROWS,)
    assert "model.layers.1.self_attn.q_proj.weight" not in out  # no unfused stragglers

    # GDN: split in_proj (fp8 qkvz + bf16 ba), fp8 out_proj
    assert out["model.layers.0.linear_attn.in_proj_qkvz.weight"].dtype == fp8
    assert out["model.layers.0.linear_attn.in_proj_qkvz.weight"].shape[0] == 2 * GDN_ROWS
    assert out["model.layers.0.linear_attn.in_proj_ba.weight"].dtype == torch.bfloat16
    assert "model.layers.0.linear_attn.in_proj.weight" not in out
    assert out["model.layers.0.linear_attn.out_proj.weight"].dtype == fp8

    # every MLP layer is native NVFP4 (packed ones AND the RTN-folded fp8 one)
    for li in (0, 1):
        for proj in ("gate_up_proj", "down_proj"):
            assert out[f"model.layers.{li}.mlp.{proj}.weight"].dtype == torch.uint8
            assert out[f"model.layers.{li}.mlp.{proj}.weight_scale"].dtype == fp8
            assert out[f"model.layers.{li}.mlp.{proj}.weight_global"].dtype == torch.float16

    # lm_head dequantized to bf16 at full width; gemma norm got its +1
    assert out["lm_head.weight"].dtype == torch.bfloat16
    assert out["model.layers.1.self_attn.q_norm.weight"].min() >= 1.0


def test_mixed_ct_rtn_fold_is_numerically_faithful(tmp_path):
    from freetoken.models.nvfp4_quant import dequant_nvfp4_reference

    tensors = _mixed_ct_tensors(mlp_fp8_layers={1})
    ckpt = _write_ckpt(tmp_path, {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": _text_config(),
        "quantization_config": _CT_MIXED_QUANT,
    }, tensors)
    out = _iter(ckpt)

    # layer-1 down_proj arrived FP8; the emitted NVFP4 must dequant back near the
    # fp8-dequantized source (only e2m1 block rounding may differ).
    key = "model.language_model.layers.1.mlp.down_proj"
    ref = (tensors[f"{key}.weight"].to(torch.float32)
           * tensors[f"{key}.weight_scale"].reshape(-1, 1).to(torch.float32))
    got = dequant_nvfp4_reference(
        out["model.layers.1.mlp.down_proj.weight"],
        out["model.layers.1.mlp.down_proj.weight_scale"],
        out["model.layers.1.mlp.down_proj.weight_global"],
        dtype=torch.float32,
    )
    assert (got - ref).abs().max() < ref.abs().max() / 6 * 1.05


def test_pure_ct_checkpoint_emission_unchanged(tmp_path):
    """Pure compressed-tensors NVFP4 (Qwen3.6-27B dialect): qkv stays packed-fused, GDN
    in_proj stays the single bf16 4-way fusion."""
    t = _mixed_ct_tensors(mlp_fp8_layers=set())
    # replace the fp8 attention with packed NVFP4 + bf16 GDN in_proj
    fp8_keys = [k for k in t if k.endswith(".weight") and t[k].dtype == torch.float8_e4m3fn]
    for k in fp8_keys:
        del t[k]
    for k in [k for k in t if k.endswith(".weight_scale")]:
        if not k.startswith("model.language_model.layers.") or ".mlp." not in k:
            del t[k]
    p0, p1 = "model.language_model.layers.0", "model.language_model.layers.1"
    for proj in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                 "linear_attn.in_proj_b", "linear_attn.in_proj_a"):
        t[f"{p0}.{proj}.weight"] = torch.randn(24, H, dtype=torch.bfloat16) * 0.05
    for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
        _, packed, scale, wg = _packed_nvfp4(Q_ROWS, H)
        base = f"{p1}.{proj}"
        t[f"{base}.weight_packed"] = packed
        t[f"{base}.weight_scale"] = scale
        t[f"{base}.weight_global_scale"] = torch.tensor([wg], dtype=torch.float32)
    w = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    t["lm_head.weight"] = w

    out = _iter(_write_ckpt(tmp_path, {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": _text_config(),
        "quantization_config": _CT_PURE_QUANT,
    }, t))

    assert out["model.layers.1.self_attn.qkv_proj.weight"].dtype == torch.uint8
    assert out["model.layers.1.self_attn.qkv_proj.weight_scale"].dtype == torch.float8_e4m3fn
    assert out["model.layers.0.linear_attn.in_proj.weight"].dtype == torch.bfloat16
    assert "model.layers.0.linear_attn.in_proj_qkvz.weight" not in out
    assert out["model.layers.1.self_attn.o_proj.weight"].dtype == torch.uint8
    assert out["lm_head.weight"].dtype == torch.bfloat16
