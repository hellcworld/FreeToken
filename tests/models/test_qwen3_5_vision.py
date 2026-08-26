"""qwen3_5_moe vision stack: numerical alignment vs the transformers reference.

The tower/merger are reimplementations of HF ``Qwen3_5VisionModel`` (CPU, fp32):
same weights -> same outputs. The interleaved-mRoPE math and the rope-index builder
are checked against the reference rotary / hand-computed positions, including the
text-only degeneration (three equal rows == plain partial rope).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.config import Qwen3_5VisionConfig
from freetoken.models.qwen3_5_moe.vision import (
    Qwen3_5PatchMerger,
    Qwen3_5VisionTower,
    apply_interleaved_mrope,
)


def _cfg() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=96,
        patch_size=8, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=128, num_position_embeddings=36,
    )


def _hf_vision_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as HFCfg

    return HFCfg(
        depth=2, hidden_size=64, num_heads=4, intermediate_size=96, patch_size=8,
        temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=128,
        num_position_embeddings=36, in_channels=3,
    )


def test_vision_tower_matches_transformers_reference():
    torch.manual_seed(3)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    ref = Qwen3_5VisionModel(_hf_vision_model()).to(torch.float32).eval()
    tower, merger = Qwen3_5VisionTower(_cfg()), Qwen3_5PatchMerger(_cfg())
    # our state-dict names must match the loader's rename exactly; values come from
    # the reference so both towers run the same weights
    ref_sd = ref.state_dict()
    ours = {
        ("embed_vision." + k[len("merger."):]) if k.startswith("merger.") else ("vision_tower." + k): v
        for k, v in ref_sd.items()
    }
    ours_names = set(tower.state_dict(prefix="vision_tower")) | set(
        merger.state_dict(prefix="embed_vision"))
    assert set(ours) == ours_names, (sorted(ours), sorted(ours_names))
    tower.load_state_dict(
        {k: v for k, v in ours.items() if k.startswith("vision_tower.")}, prefix="vision_tower"
    )
    merger.load_state_dict(
        {k: v for k, v in ours.items() if k.startswith("embed_vision.")}, prefix="embed_vision"
    )

    grid = torch.tensor([[1, 8, 8], [1, 4, 8]])  # two images, ragged grids
    n_patches = sum(t * h * w for t, h, w in grid.tolist())
    pixels = torch.randn(n_patches, 3 * 2 * 8 * 8, dtype=torch.float32)
    with torch.no_grad():
        ref_out = ref.forward(pixels, grid_thw=grid)
        ours_out = merger.forward(tower.forward(pixels, grid))
    torch.testing.assert_close(ours_out, ref_out.pooler_output, rtol=2e-4, atol=2e-4)


def test_interleaved_mrope_matches_reference_rotary():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        hidden_size=256, num_attention_heads=4, head_dim=256, num_key_value_heads=2,
        rope_parameters={"rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                          "mrope_section": [11, 11, 10]},
    )
    ref_rot = Qwen3_5TextRotaryEmbedding(config=cfg)
    positions = torch.randint(0, 500, (3, 1, 64))
    x = torch.zeros(1, 64, 4, 256)  # (bs, seq, heads, head_dim) shape carrier
    ref_cos, ref_sin = ref_rot.forward(x, positions)
    ref_cos, ref_sin = ref_cos[0], ref_sin[0]  # [64, 64]

    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = positions.float().squeeze(1).unsqueeze(-1) * inv_freq  # [3, 64, 32]
    merged = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = [11, 11, 10][axis] * 3
        merged[:, offset:length:3] = freqs[axis][:, offset:length:3]
    emb = torch.cat((merged, merged), dim=-1)
    torch.testing.assert_close(emb.cos(), ref_cos.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(emb.sin(), ref_sin.float(), rtol=1e-5, atol=1e-5)


def test_interleaved_mrope_text_rows_degenerate_to_plain_rope():
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    pos = torch.arange(40)
    mm = pos.unsqueeze(0).expand(3, -1).contiguous()
    q = torch.randn(40, 2, 256, dtype=torch.float32)
    k = torch.randn(40, 1, 256, dtype=torch.float32)
    q_m, k_m = apply_interleaved_mrope(mm, inv_freq, (11, 11, 10), q, k)
    # plain partial rope on the first 64 dims
    freqs = pos.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def _rot(t):
        a, b = t.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    q_p = torch.cat((q[..., :64] * cos + _rot(q[..., :64]) * sin, q[..., 64:]), dim=-1)
    torch.testing.assert_close(q_m, q_p, rtol=1e-6, atol=1e-6)


def test_rope_index_builder_mixed_prompt():
    from freetoken.models.qwen3_5_moe.vision import qwen3_5_rope_index

    img, vs, ve = 248056, 248053, 248054
    grid = torch.tensor([[1, 4, 4]])  # 2x2 merged tokens
    ids = torch.tensor([1, 2, 3, vs, img, img, img, img, ve, 9])
    rows, delta = qwen3_5_rope_index(ids, grid, img, vs, ve)
    assert rows.shape == (3, 10)
    # text 0..2, vision_start takes 3, image span starts at 4 (T fixed, H/W raster),
    # advance max(2,2)=2 -> vision_end at 6, trailing text at 7
    assert rows[0].tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 6, 7]
    assert rows[1].tolist() == [0, 1, 2, 3, 4, 4, 5, 5, 6, 7]
    assert rows[2].tolist() == [0, 1, 2, 3, 4, 5, 4, 5, 6, 7]
    assert delta == 7 + 1 - 10  # -2


def test_rope_index_builder_text_only():
    from freetoken.models.qwen3_5_moe.vision import qwen3_5_rope_index

    ids = torch.arange(10) + 300
    rows, delta = qwen3_5_rope_index(ids, torch.zeros(0, 3, dtype=torch.int64), 248056)
    assert rows.shape == (3, 10)
    assert torch.equal(rows[0], rows[1]) and torch.equal(rows[0], rows[2])
    assert rows[0].tolist() == list(range(10)) and delta == 0
