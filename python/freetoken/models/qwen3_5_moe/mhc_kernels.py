"""Fused mHC router split + Sinkhorn normalization (SenseNova-Loop hyper-connections).

Per token the router projects its RMS-normalized flattened residual streams with an
fp32 linear to ``[(2+streams)*streams]`` logits; the split produces ``pre`` (sigmoid),
``post`` (2*sigmoid) and ``mix`` (``iterations`` of alternating row/column log-space
normalization, then exp -- the SenseNova-Loop schedule, with NO softmax init and NO
eps, unlike the DeepSeek-V4 variant). In eager torch that is ~120 tiny reductions per
router call x 96 router calls per token -- tens of thousands of launches. This
collapses the split into one launch: one program per token, the ``streams x streams``
matrix lives in registers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_split_kernel(
    logits_ptr, pre_ptr, post_ptr, mix_ptr, n,
    stride_ln, stride_pn, stride_pon, stride_mn,
    S: tl.constexpr, ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return
    s = tl.arange(0, S)
    lg = logits_ptr + row * stride_ln

    pre = tl.sigmoid(tl.load(lg + s))
    tl.store(pre_ptr + row * stride_pn + s, pre)

    post = 2.0 * tl.sigmoid(tl.load(lg + S + s))
    tl.store(post_ptr + row * stride_pon + s, post)

    idx = s[:, None] * S + s[None, :]  # [S, S] row-major within the mix block
    c = tl.load(lg + 2 * S + idx)
    # iterations x (row logsumexp, column logsumexp) in log space, subtracting the
    # running max inside each reduction for stability (== torch.logsumexp exactly)
    for _ in range(ITERS):
        rmax = tl.max(c, axis=1)
        c = c - (rmax + tl.log(tl.sum(tl.exp(c - rmax[:, None]), axis=1)))[:, None]
        cmax = tl.max(c, axis=0)
        c = c - (cmax + tl.log(tl.sum(tl.exp(c - cmax[None, :]), axis=0)))[None, :]
    tl.store(mix_ptr + row * stride_mn + idx, tl.exp(c))


def mhc_split_sinkhorn(logits: torch.Tensor, streams: int, iterations: int):
    """``logits`` ``[n, (2+streams)*streams]`` fp32 -> ``(pre[n,s], post[n,s], mix[n,s,s])``
    fp32. CUDA fast path is the fused kernel; the torch fallback IS the HF reference
    math (bit-identical schedule) for CPU hosts and kernel-less environments."""
    n = logits.shape[0]
    s = streams
    dev = logits.device
    if dev.type == "cuda":
        logits = logits.contiguous()
        pre = torch.empty(n, s, device=dev, dtype=torch.float32)
        post = torch.empty(n, s, device=dev, dtype=torch.float32)
        mix = torch.empty(n, s, s, device=dev, dtype=torch.float32)
        _mhc_split_kernel[(n,)](
            logits, pre, post, mix, n,
            logits.stride(0), pre.stride(0), post.stride(0), mix.stride(0),
            S=s, ITERS=iterations, num_warps=1,
        )
        return pre, post, mix
    # reference: the SenseNova-Loop MHCRouter.forward split, verbatim
    pre = torch.sigmoid(logits[..., :s])
    post = 2 * torch.sigmoid(logits[..., s : 2 * s])
    c = logits[..., 2 * s :].unflatten(-1, (s, s)).float()
    for _ in range(iterations):
        c = c - torch.logsumexp(c, dim=-1, keepdim=True)
        c = c - torch.logsumexp(c, dim=-2, keepdim=True)
    return pre, post, c.exp()


__all__ = ["mhc_split_sinkhorn"]
