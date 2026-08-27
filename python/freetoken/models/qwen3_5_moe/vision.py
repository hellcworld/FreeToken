"""Qwen3.5/3.6/3.8 vision tower + patch merger (the ``model.visual.*`` stack).

Bit-faithful reimplementation of the HF ``Qwen3_5VisionModel`` reference (a
Qwen-VL-style ViT): Conv3d patch embed (kernel=stride=``[temporal, patch, patch]``,
equivalently a linear over flattened ``[C, T, P, P]`` patch rows), a learnable
square position table bilinearly resampled (align_corners=True) to each image's
patch grid, depth x [LayerNorm -> fused-QKV bidirectional attention (per-frame
segments) with 2-D rotary (h|w frequencies) -> LayerNorm -> tanh-GELU MLP], and
the PatchMerger (pre-merge LayerNorm -> 2x2 spatial merge -> fc1 -> erf-GELU ->
fc2 -> text hidden). No DeepStack (Qwen3.5 removed it).

State-dict keys match the checkpoint exactly (``blocks.N.{norm1,attn.qkv,attn.proj,
norm2,mlp.linear_fc1,mlp.linear_fc2}``, ``patch_embed.proj``, ``pos_embed``); the
merger is mounted as the model's ``embed_vision`` (``FREETOKEN_LOAD_VISION`` opt-in).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, OPList

if TYPE_CHECKING:
    from .config import Qwen3_5VisionConfig


class VisionLinear(BaseOP):
    """``nn.Linear`` as a BaseOP (weight/bias attrs). The patch-embed ``proj`` keeps its
    5-D Conv3d weight; ``forward`` views it 2-D so the conv-by-stride == linear math."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.empty(out_features) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.view(self.weight.shape[0], -1)
        return F.linear(x, w, self.bias)


class VisionLayerNorm(BaseOP):
    def __init__(self, hidden: int, eps: float = 1e-6):
        self.weight = torch.empty(hidden)
        self.bias = torch.empty(hidden)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self._eps)


class VisionEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


# ------------------------------------------------------------------ grid helpers
def vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """``(total_patches, 2)`` [h, w] patch coords in spatial-merge-block order, the
    h/w pair repeated ``t`` times per frame (the HF ``get_vision_position_ids`` layout:
    block-major ``(h//m, m, w//m, m)`` transpose)."""
    out = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )
        m = spatial_merge_size
        block_shape = (h // m, m, w // m, m)
        hpos = hpos.reshape(block_shape).transpose(1, 2).flatten()
        wpos = wpos.reshape(block_shape).transpose(1, 2).flatten()
        out.append(torch.stack((hpos, wpos), dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0).to(grid_thw.device)


def vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Cumulative per-FRAME sequence boundaries (each frame attends bidirectionally to
    itself only; ``h * w`` per frame, ``t`` segments per entry)."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def pos_embed_interp(grid_thw: torch.Tensor, num_grid_per_side: int,
                     spatial_merge_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch (4-tap) gather indices/weights resampling the square
    ``num_grid_per_side`` position table to each image's (h, w) grid, bilinear with
    align_corners=True (the closed form of ``linspace(0, side-1, size)[index]``).
    Patches are enumerated in spatial-merge-block order, matching the tower input."""
    side, merge = num_grid_per_side, spatial_merge_size
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(int(counts.sum()), device=starts.device) - starts) % (heights * widths)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    def _axis(index: torch.Tensor, size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = index.to(torch.float32) * (side - 1) / torch.clamp(size - 1, min=1)
        floor = torch.floor(src)
        taps = (floor.long().unsqueeze(-1) + torch.arange(2, device=index.device)).clamp(0, side - 1)
        dist = (src.unsqueeze(-1) - floor.unsqueeze(-1) - torch.arange(2, device=index.device)).abs()
        return taps, (1 - dist).clamp(min=0)

    h_taps, h_w = _axis(row, heights)
    w_taps, w_w = _axis(col, widths)
    indices = (h_taps.unsqueeze(-1) * side + w_taps.unsqueeze(1)).reshape(-1, 4)
    weights = (h_w.unsqueeze(-1) * w_w.unsqueeze(1)).reshape(-1, 4)
    return indices, weights


# ---------------------------------------------------------------------- blocks
class _VisionRotary:
    """2-D vision rotary: 18 h-frequencies ++ 18 w-frequencies -> 36, mirrored to 72
    (half-split rotate). No parameters, computed per batch."""

    def __init__(self, half_dim: int, base: float = 10000.0):
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
        )
        # the model is built under a meta-device context, which leaves this
        # computed constant dataless; recompute lazily on first real use
        self._base = base
        self._half_dim = half_dim

    def __call__(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.inv_freq.is_meta:
            self.inv_freq = 1.0 / (
                self._base ** (torch.arange(
                    0, self._half_dim, 2, dtype=torch.float32, device=position_ids.device
                ) / self._half_dim)
            )
        freqs = torch.cat(
            [position_ids[:, 0:1].float() * self.inv_freq.to(position_ids.device),
             position_ids[:, 1:2].float() * self.inv_freq.to(position_ids.device)],
            dim=-1,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class VisionAttention(BaseOP):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        hidden = cfg.hidden_size
        self.qkv = VisionLinear(hidden, hidden * 3, bias=True)
        self.proj = VisionLinear(hidden, hidden, bias=True)
        self._num_heads = cfg.num_heads
        self._head_dim = cfg.head_dim

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seq = x.shape[0]
        qkv = self.qkv.forward(x).view(seq, 3, self._num_heads, self._head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        # rope in fp32 (reference casts up), then back
        cos1, sin1 = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
        q = (q.float() * cos1 + _rotate_half(q.float()) * sin1).to(x.dtype)
        k = (k.float() * cos1 + _rotate_half(k.float()) * sin1).to(x.dtype)
        # per-frame bidirectional attention: [1, H, L, D] per segment
        outs = []
        for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            qs = q[s:e].transpose(0, 1).unsqueeze(0)
            ks = k[s:e].transpose(0, 1).unsqueeze(0)
            vs = v[s:e].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            outs.append(o.squeeze(0).transpose(0, 1).reshape(e - s, -1))
        return self.proj.forward(torch.cat(outs, dim=0))


class VisionMLP(BaseOP):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        self.linear_fc1 = VisionLinear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = VisionLinear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(
            F.gelu(self.linear_fc1.forward(x), approximate="tanh")
        )


class VisionBlock(BaseOP):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        self.norm1 = VisionLayerNorm(cfg.hidden_size)
        self.attn = VisionAttention(cfg)
        self.norm2 = VisionLayerNorm(cfg.hidden_size)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, cu_seqlens, cos, sin):
        x = x + self.attn.forward(self.norm1.forward(x), cu_seqlens, cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen3_5VisionTower(BaseOP):
    """The ``model.visual.{patch_embed,pos_embed,blocks}`` stack; emits per-patch
    hidden states (NOT merged) in merge-block order."""

    def __init__(self, cfg: Qwen3_5VisionConfig):
        self.patch_embed = _PatchEmbed(cfg)
        self.pos_embed = VisionEmbedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = OPList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self._num_grid_per_side = int(math.isqrt(cfg.num_position_embeddings))
        self._spatial_merge_size = cfg.spatial_merge_size
        self._rotary = _VisionRotary(cfg.head_dim // 2)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values``: [total_patches, C*T*P*P]; ``grid_thw``: [n, 3]."""
        x = self.patch_embed.forward(pixel_values)
        indices, weights = pos_embed_interp(
            grid_thw, self._num_grid_per_side, self._spatial_merge_size
        )
        pos = (self.pos_embed.forward(indices) * weights.unsqueeze(-1)).sum(1)
        x = x + pos.to(x.dtype)
        position_ids = vision_position_ids(grid_thw, self._spatial_merge_size)
        cos, sin = self._rotary(position_ids)
        cu = vision_cu_seqlens(grid_thw)
        for blk in self.blocks.op_list:
            x = blk.forward(x, cu, cos, sin)
        return x


class _Conv3dPatchProj(BaseOP):
    """The patch-embed Conv3d stored verbatim (5-D ``[hidden, C, T, P, P]``) and applied
    as a linear over flattened patch rows (kernel == stride makes them equivalent)."""

    def __init__(self, cfg: Qwen3_5VisionConfig):
        self.weight = torch.empty(
            cfg.hidden_size, cfg.in_channels,
            cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size,
        )
        self.bias = torch.empty(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.view(self.weight.shape[0], -1), self.bias)


class _PatchEmbed(BaseOP):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        self.proj = _Conv3dPatchProj(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(x)


class Qwen3_5PatchMerger(BaseOP):
    """The ``model.visual.merger`` stack, mounted as the model's ``embed_vision``:
    pre-merge LayerNorm -> 2x2 spatial merge (adjacent merge-block-order patches) ->
    fc1 -> erf-GELU -> fc2 -> text hidden."""

    def __init__(self, cfg: Qwen3_5VisionConfig):
        merged = cfg.hidden_size * cfg.spatial_merge_size**2
        self.norm = VisionLayerNorm(cfg.hidden_size)
        self.linear_fc1 = VisionLinear(merged, merged, bias=True)
        self.linear_fc2 = VisionLinear(merged, cfg.out_hidden_size, bias=True)
        self._merged = merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm.forward(x).view(-1, self._merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


def apply_interleaved_mrope(
    mm_positions: torch.Tensor, inv_freq: torch.Tensor,
    section: tuple[int, int, int], q: torch.Tensor, k: torch.Tensor,
):
    """Interleaved mRoPE per the HF reference: per-axis freqs [3, N, n_freq], then
    frequency slot i of the merged row takes axis i%3 (sections [T, H, W], frequencies
    continuous). Text tokens carry three equal rows, which degenerates to plain partial
    rope. fp32 math, cast back on return."""
    mm_positions = mm_positions.to(inv_freq.device)
    freqs = mm_positions.float().unsqueeze(-1) * inv_freq  # [3, N, 32]
    merged = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = section[axis] * 3
        merged[:, offset:length:3] = freqs[axis][:, offset:length:3]
    emb = torch.cat((merged, merged), dim=-1)  # [N, rotary_dim]
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)  # [N, 1, rotary_dim]
    rdim = emb.shape[-1]
    q_rot, k_rot = q[..., :rdim].float(), k[..., :rdim].float()

    def _rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q = torch.cat(((q_rot * cos + _rot(q_rot) * sin).to(q.dtype), q[..., rdim:]), dim=-1)
    k = torch.cat(((k_rot * cos + _rot(k_rot) * sin).to(k.dtype), k[..., rdim:]), dim=-1)
    return q, k


__all__ = ["Qwen3_5VisionTower", "Qwen3_5PatchMerger", "apply_interleaved_mrope",
           "vision_position_ids", "vision_cu_seqlens", "pos_embed_interp", "qwen3_5_rope_index"]


def qwen3_5_rope_index(
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    image_token_id: int,
    vision_start_token_id: int = 248053,
    vision_end_token_id: int = 248054,
) -> tuple[torch.Tensor, int]:
    """Interleaved-mRoPE (T, H, W) rows for a prompt with image placeholders, plus the
    decode ``rope_delta`` -- the HF ``get_rope_index`` semantics for image inputs:

    * text segments: all three rows are ``arange(len) + current_pos``
    * an image span (vision_start..vision_end, filled with ``image_token_id``):
      T/H/W coordinates of the merged (h//2 x w//2) grid, each offset by
      ``current_pos`` (T after the per-frame arange), tokens in frame-major raster
      order (the tower's merge-block output order); the span advances
      ``current_pos`` by ``max(h, w) // 2``
    * ``delta = positions.max() + 1 - seq_len`` shifts all later decode positions

    Video spans (video_token_id with per-frame timestamps) are not supported yet.
    """
    ids = input_ids.tolist()
    grids = [tuple(int(v) for v in g) for g in grid_thw]
    rows = [[], [], []]  # T, H, W
    pos = 0
    i = 0
    gi = 0
    while i < len(ids):
        if ids[i] == vision_start_token_id:
            rows[0].append(pos); rows[1].append(pos); rows[2].append(pos)  # start: text
            pos += 1
            j = ids.index(vision_end_token_id, i)
            span = j - i - 1  # placeholder count
            t, h, w = grids[gi]
            gi += 1
            t_m, h_m, w_m = t, h // 2, w // 2
            assert t_m * h_m * w_m == span, (
                f"image span ({span} tokens) != grid {t}x{h//2}x{w//2}"
            )
            for tt in range(t_m):
                for hh in range(h_m):
                    for ww in range(w_m):
                        rows[0].append(pos + tt)
                        rows[1].append(pos + hh)
                        rows[2].append(pos + ww)
            pos += max(h_m, w_m)  # reference advance: max(h, w) // 2 patches
            rows[0].append(pos); rows[1].append(pos); rows[2].append(pos)  # end: text
            pos += 1
            i = j + 1
        else:
            rows[0].append(pos)
            rows[1].append(pos)
            rows[2].append(pos)
            pos += 1
            i += 1
    positions = torch.tensor([rows], dtype=torch.int64).squeeze(0)  # [3, seq]
    delta = int(positions.max()) + 1 - len(ids)
    return positions, delta
