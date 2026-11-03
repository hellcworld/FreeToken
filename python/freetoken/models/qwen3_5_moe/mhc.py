"""Manifold-constrained hyper-connections (mHC) for SenseNova-Loop's looped layers.

Mirrors the official remote-code ``MHCRouter`` exactly: per layer two fp32 routers
(``mhc_attn`` before the mixer, ``mhc_ffn`` before the MLP) read the flattened
``streams`` residual streams, RMS-normalize (plain, unweighted -- the routing decision
only), and project to the routing logits. ``pre`` collapses the streams into the
sublayer's single input branch; ``post`` re-expands the branch output into every
stream; ``mix`` (Sinkhorn-doubly-stochastic) blends the old streams. The stream
EXPAND (broadcast copy of the incoming hidden into every stream) happens at the mHC
range's first physical layer and the REDUCE (mean over streams) at its last -- both
re-run on every loop visit.

The tiny fp32 router GEMM and the sigmoid/Sinkhorn split run through
:mod:`freetoken.models.qwen3_5_moe.mhc_kernels`; the stream combine/re-expand reuse
the DeepSeek-V4 fused hc kernels (``hc_pre_combine`` / ``hc_post_combine`` -- same
math; the DSV4 ``comb[p,q]`` indexes the transpose of this model's ``mix[q,p]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP

from .mhc_kernels import mhc_split_sinkhorn

if TYPE_CHECKING:
    from freetoken.models.config import LoopSettings


class MHCRouter(BaseOP):
    """One mHC routing point (``mhc_attn`` or ``mhc_ffn`` on a looped layer).

    fp32 parameters by construction: the engine's weight mount casts to the module's
    declared dtype, so the fp32 ``weight``/``bias`` buffers below survive the model's
    bf16 downcast exactly like GDN's ``A_log``/``dt_bias``.
    """

    def __init__(self, hidden_size: int, settings: LoopSettings):
        s = settings.streams
        self.streams = s
        self.iterations = settings.iterations
        self.fixed_identity = settings.fixed_identity
        self.pre_scale = settings.pre_scale
        out_size = 2 * s + (0 if settings.fixed_identity else s * s)
        self.weight = torch.empty(out_size, s * hidden_size, dtype=torch.float32)
        self.bias = torch.empty(out_size, dtype=torch.float32)

    def route(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``residual`` [N, streams, D] -> ``(branch [N, D], post [N, s] fp32,
        mix [N, s, s] fp32)``. ``branch`` is the pre-weighted stream sum the sublayer
        consumes (its layernorm happens in the decoder layer, matching HF)."""
        flat = residual.flatten(-2).float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(normalized, self.weight, self.bias)
        if self.fixed_identity:
            pre = torch.sigmoid(logits[..., : self.streams])
            post = 2 * torch.sigmoid(logits[..., self.streams : 2 * self.streams])
            mix = (
                torch.eye(self.streams, device=residual.device, dtype=torch.float32)
                .expand(residual.shape[0], self.streams, self.streams)
                .contiguous()
            )
        else:
            pre, post, mix = mhc_split_sinkhorn(logits, self.streams, self.iterations)
        if residual.is_cuda:
            from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

            # cast the coefficients to the stream dtype first so the fused (fp32-math)
            # kernel reproduces HF's bf16 coefficient rounding
            branch = hc_pre_combine(
                residual, pre.to(residual.dtype), residual.dtype
            )
        else:
            branch = torch.matmul(
                pre.to(residual.dtype).unsqueeze(-2), residual
            ).squeeze(-2)
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch, post, mix


def mhc_merge(
    residual: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    """HF ``MHCRouter.merge``: ``out[q, d] = sum_p mix[q, p] * residual[p, d] +
    post[q] * branch[d]`` -- the sublayer output re-expands into every stream."""
    if residual.is_cuda:
        from freetoken.kernel.triton.dsv4.hc import hc_post_combine

        # DSV4's comb indexes comb[p, q] (in-stream, out-stream) == mix.T
        comb = mix.transpose(-1, -2).to(branch.dtype).contiguous()
        return hc_post_combine(branch, residual, post.to(branch.dtype), comb)
    mixed = torch.matmul(mix.to(residual.dtype), residual)
    return mixed + post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)


__all__ = ["MHCRouter", "mhc_merge"]
