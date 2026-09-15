from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    detect_compressed_tensors_mixed_fp8,
    detect_compressed_tensors_nvfp4,
    vision_load_enabled,
)


@dataclass(frozen=True)
class Qwen3_5VisionConfig:
    """The Qwen3.5/3.6/3.8 ``vision_config`` (a Qwen-VL-style ViT + patch merger).

    Field names mirror the HF ``Qwen3_5VisionConfig``; ``hidden_size`` is the tower's
    width (1152 at 27B/35B), ``out_hidden_size`` the text hidden the merger emits.
    """

    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    in_channels: int = 3

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _parse_vision_config(hf_config: Any) -> Qwen3_5VisionConfig | None:
    """Text-only unless ``FREETOKEN_LOAD_VISION`` is set (same opt-in as gemma4): the
    tower + merger are ~0.9 GiB of never-quantized bf16 weights that text-only serving
    never touches. Returns None for a text-only build or when vision is disabled."""
    vc = getattr(hf_config, "vision_config", None)
    if vc is None or not vision_load_enabled():
        return None
    return Qwen3_5VisionConfig(
        depth=vc.depth,
        hidden_size=vc.hidden_size,
        num_heads=vc.num_heads,
        intermediate_size=vc.intermediate_size,
        patch_size=vc.patch_size,
        temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
        spatial_merge_size=getattr(vc, "spatial_merge_size", 2),
        out_hidden_size=vc.out_hidden_size,
        num_position_embeddings=vc.num_position_embeddings,
        in_channels=getattr(vc, "in_channels", 3),
    )


def _modelopt_quant_file(hf_config: Any) -> dict | None:
    """modelopt checkpoints ALSO ship their quant config in ``hf_quant_config.json``
    (under ``quantization``), mirrored into config.json's ``quantization_config`` by the
    nvidia export pipeline. Exports that strip the mirror (the Desktop catalog builds)
    then parse as *unquantized* and misroute into the pure-NVFP4 loader, which
    dequantizes every tensor through the GPU kernel at load -- the #139 failure mode.
    Returns the file's ``quantization`` dict, or None when there is no such file."""
    path = getattr(hf_config, "name_or_path", None) or getattr(hf_config, "_name_or_path", None)
    if not isinstance(path, str) or not path:
        return None
    key = os.path.abspath(path) if os.path.isdir(path) else path
    return _modelopt_quant_file_cached(key)


@functools.cache
def _modelopt_quant_file_cached(key: str) -> dict | None:
    def _read(fp: str) -> dict | None:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        quant = data.get("quantization") if isinstance(data, dict) else None
        return quant if isinstance(quant, dict) and quant else None

    if os.path.isfile(os.path.join(key, "hf_quant_config.json")):
        return _read(os.path.join(key, "hf_quant_config.json"))
    if os.path.isdir(key):  # local dir without the file
        return None
    try:  # hub id: small file, cached by huggingface_hub after the first fetch
        from huggingface_hub import hf_hub_download

        return _read(hf_hub_download(key, "hf_quant_config.json"))
    except Exception:  # noqa: BLE001 -- no file / offline -> detection stays config-only
        return None


def _quant_accessor(hf_config: Any):
    """A ``get(key, default=None)`` accessor over the HF ``quantization_config`` (dict or
    object), or ``None`` when the model has no quant config.

    Keys missing from config.json's copy fall back to ``hf_quant_config.json`` (the
    modelopt source of truth), so a stripped export still detects like its mirrored
    sibling; config.json always wins per key."""
    quant = getattr(hf_config, "quantization_config", None)
    file_quant = _modelopt_quant_file(hf_config) if quant is None or not _has_modelopt_fields(quant) else None
    if quant is None and file_quant is None:
        return None
    merged = dict(file_quant or {})
    if quant is not None:
        if isinstance(quant, dict):
            merged.update({k: v for k, v in quant.items() if v is not None})
        else:
            for k in ("quant_method", "quant_algo", "quantized_layers", "weight_block_size"):
                v = getattr(quant, k, None)
                if v is not None:
                    merged[k] = v
    return merged.get


def _has_modelopt_fields(quant: Any) -> bool:
    """Does config.json's quantization_config already carry the modelopt fields the
    detectors read? (``quant_algo`` / ``quantized_layers``.)"""
    if isinstance(quant, dict):
        return quant.get("quant_algo") is not None or quant.get("quantized_layers") is not None
    return (
        getattr(quant, "quant_algo", None) is not None
        or getattr(quant, "quantized_layers", None) is not None
    )


def _fp8_block_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """Detect DeepSeek-V3-style 128x128 block-fp8 from HF ``quantization_config``.

    Returns ``("fp8_block", (block_n, block_k))`` for a block-fp8 checkpoint (weights
    fp8-e4m3 + per-block ``weight_scale_inv``, dynamic activation), else ``("none", None)``.
    The quantization_config sits on the top-level hf_config (not ``text_config``).
    """
    get = _quant_accessor(hf_config)
    if get is None:
        return "none", None
    method = str(get("quant_method") or get("quant_algo") or "").lower()
    block = get("weight_block_size")
    if method == "fp8" and block:
        bs = tuple(int(x) for x in block)
        assert bs == (128, 128), f"only 128x128 block-fp8 is supported, got {bs}"
        return "fp8_block", bs
    return "none", None


def _expert_quant(hf_config: Any) -> str:
    """Quantization format of the *routed* experts (the only weights served from the
    offload cache). The nvidia/modelopt checkpoints are either plain NVFP4 (``quant_algo``
    ``NVFP4``) or ``MIXED_PRECISION`` (per-layer ``quantized_layers`` map); in the mixed
    case the routed experts carry their own ``W4A16_NVFP4``/``FP8`` algo. Dense quantized
    weights (attention/shared-expert/lm_head) are handled separately by dequant-at-load."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    algo = str(get("quant_algo") or get("quant_method") or "").lower()
    if "fp4" in algo:
        return "nvfp4"
    if "mixed" in algo:
        layers = get("quantized_layers") or {}
        for name, spec in (layers.items() if isinstance(layers, dict) else []):
            if name.endswith(".mlp.experts") or ".mlp.experts." in name:
                expert_algo = str((spec or {}).get("quant_algo", "")).lower()
                if "fp4" in expert_algo:
                    return "nvfp4"
                if "fp8" in expert_algo:
                    return "fp8"
    return "none"


# Detection now lives in models/config.py (shared with muse_glimmer); weight.py imports
# it under this name.
_compressed_tensors_nvfp4 = detect_compressed_tensors_nvfp4


def _lm_head_quant(hf_config: Any) -> str:
    """Whether the checkpoint stores ``lm_head`` as NVFP4. modelopt MIXED_PRECISION lists it in
    the per-layer ``quantized_layers`` map (``W4A16_NVFP4``); pure-NVFP4 checkpoints have no
    per-layer map and leave lm_head bf16. Returns ``"nvfp4"`` or ``"none"``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name == "lm_head" or name.endswith(".lm_head"):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _dense_mlp_quant(hf_config: Any) -> str:
    """NVFP4 on the *dense* (non-MoE) decoder MLP. modelopt MIXED_PRECISION dense checkpoints
    (e.g. Qwen3.6-27B-NVFP4) list ``.mlp.{gate,up,down}_proj`` as ``W4A16_NVFP4`` in
    ``quantized_layers``; MoE checkpoints have ``.mlp.experts.*`` / ``.mlp.shared_expert.*``
    instead (covered by ``expert_quant``). ``endswith(".mlp.gate_proj")`` matches only the bare
    dense MLP -- not ``.mlp.shared_expert.gate_proj`` nor ``.mlp.experts.N.gate_proj``."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        if name.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")):
            if "fp4" in str((spec or {}).get("quant_algo", "")).lower():
                return "nvfp4"
    return "none"


def _attn_quant(hf_config: Any) -> str:
    """Per-tensor FP8 on the *dense* attention/GDN projections. The modelopt
    ``MIXED_PRECISION`` checkpoints tag ``self_attn.{q,k,v,o}_proj`` and
    ``linear_attn.{in_proj_qkv,in_proj_z,out_proj}`` with ``quant_algo`` ``FP8`` (fp8-e4m3
    weight + a scalar ``weight_scale``; W8A16). Returns ``"fp8_pertensor"`` when present,
    else ``"none"`` (NVFP4 dense weights -- shared_expert/lm_head -- stay dequant-at-load)."""
    get = _quant_accessor(hf_config)
    if get is None:
        return "none"
    layers = get("quantized_layers") or {}
    if not isinstance(layers, dict):
        return "none"
    for name, spec in layers.items():
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8" and (".self_attn." in name or ".linear_attn." in name):
            return "fp8_pertensor"
    return "none"


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = getattr(text, "rope_parameters", None) or {}
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = round(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    rope_type = rope_params.get("rope_type", "default")
    rope_scaling = (
        None
        if rope_type in (None, "default")
        else {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
    )

    expert_quant, weight_block_size = _fp8_block_quant(hf_config)
    if expert_quant == "none":
        expert_quant = _expert_quant(hf_config)  # nvfp4 / mixed-precision modelopt
    # Dense attention/GDN quant is independent of the routed experts (block-fp8 already
    # quantizes both, so only probe for per-tensor FP8 when experts aren't block-fp8).
    attn_quant = "none" if expert_quant == "fp8_block" else _attn_quant(hf_config)
    # NVFP4 checkpoints store the dense MLP projections (shared_expert; dense non-MoE MLP) as
    # packed FP4 exactly like the routed experts -- independent of whether attention is FP8
    # (mixed) or bf16 (pure NVFP4). Keep them native FP4 (W4A16) whenever the experts are
    # NVFP4. The lm_head is detected separately (only the mixed checkpoint quantizes it).
    # MoE-NVFP4 keeps the shared_expert dense MLP native FP4 (expert_quant=="nvfp4"); a dense
    # (non-MoE) modelopt checkpoint instead tags the bare .mlp.{gate,up,down}_proj as NVFP4.
    dense_quant = "nvfp4" if expert_quant == "nvfp4" else _dense_mlp_quant(hf_config)
    lm_head_quant = _lm_head_quant(hf_config)

    # compressed-tensors NVFP4 (dense Qwen3.6-27B): the attention (q/k/v/o, GDN out_proj) AND
    # the dense MLP are W4A16 NVFP4; GDN in_proj_*, lm_head, norms stay bf16. Wire the shared
    # W4A16 kernels (attn_quant=="nvfp4" routes the attention/GDN linears through them too).
    # Mixed-precision CT export (an 8-bit float group alongside the NVFP4 one, e.g.
    # unsloth/Qwen3.8-27B-NVFP4): the FP8 per-channel projections (attention/GDN/lm_head,
    # trailing dense-MLP layers) serve on the per-tensor-FP8 W8A16 path instead -- a
    # per-channel weight scale IS the per-output-row scale those kernels consume.
    if _compressed_tensors_nvfp4(hf_config):
        mixed_fp8 = detect_compressed_tensors_mixed_fp8(hf_config)
        attn_quant = "fp8_pertensor" if mixed_fp8 else "nvfp4"
        dense_quant = "nvfp4"
        lm_head_quant = "none"
        # Debug escape hatch: force the whole dense stack to bf16 (dequant-at-load) to
        # isolate the W4A16 kernels on untested hardware.
        import os as _os

        if _os.environ.get("FREETOKEN_CT_DENSE_BF16") == "1":
            attn_quant = "none"
            dense_quant = "none"
        # A ct MoE export (e.g. lyf/Qwen3.6-35B-A3B-*-NVFP4) packs its routed experts
        # NVFP4 exactly like the modelopt checkpoints -- the bank loader needs the
        # "nvfp4" provider, not bf16. Dense ct checkpoints (num_experts==0) stay "none".
        if int(getattr(text, "num_experts", 0) or 0) > 0:
            expert_quant = "nvfp4"

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    mrope_section = rope_params.get("mrope_section")
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=text.max_position_embeddings,
        base=rope_theta,
        scaling=rope_scaling,
        mrope_section=tuple(mrope_section) if mrope_section else None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate=True,
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=bool(getattr(text, "norm_topk_prob", False)),
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=_parse_vision_config(hf_config),
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        attn_quant=attn_quant,
        dense_quant=dense_quant,
        lm_head_quant=lm_head_quant,
        # Trailing MTP draft head (DeepSeek-style: an eh_proj + one full-attention decoder
        # layer sharing the target embedding/lm_head). The weights stay dropped at load
        # unless the engine enables speculation (enable_mtp); only that path builds the
        # draft module, so text-only serving never pays its VRAM.
        num_mtp_layers=int(getattr(text, "mtp_num_hidden_layers", 0) or 0),
    )


__all__ = ["parse_config"]
