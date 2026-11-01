"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swigluoai_and_mul,
)
from freetoken.moe.fused import moe_align_block_size

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _run_act(
    activation: str,
    gate_up: torch.Tensor,
    out: torch.Tensor,
    act_alpha: float,
    act_limit: float,
) -> None:
    """gemm1 -> gemm2 activation dispatch. ``swigluoai`` (MiniMax-M3, clamped
    gpt-oss swiglu over the banks' uninterleaved [gate; up] halves) carries the
    per-model ``act_alpha``/``act_limit`` scalars; the plain *_and_mul kinds
    ignore them."""
    if activation == "swigluoai":
        swigluoai_and_mul(gate_up, out, alpha=act_alpha, limit=act_limit)
        return
    _ACT[activation](gate_up, out)

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode config (int32 wide loads + deferred reduction + pair dequant).
# Offline sweep over the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16,
# BLOCK_KW=16 (== 128 k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM
# ~43%->~51% of peak BW.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_WARPS = 4

# K-major banks (nvfp4_t): N-contiguous weight tiles make wide BLOCK_N the winner
# (the same sweep on the native layout had bn32 60% SLOWER than bn16). RTX 3060
# sweep: bn64/kw32/w4 beats the native config on every decode shape (up to 1.8x on
# the K=768 gemm2, 282-295 GB/s ~ 82% peak on the H=4096 shapes).
_DECODE_MARLIN_KM_BLOCK_N = 64
_DECODE_MARLIN_KM_BLOCK_KW = 32
_DECODE_MARLIN_KM_WARPS = 4


def _bank_geometry(packed: torch.Tensor, scale: torch.Tensor):
    """Kernel geometry for either bank layout. Returns
    ``(N, K, packed_i32, (pe, pn, pkw), (se, sn, sblk))`` with the stride tuples in
    kernel-argument order.

    Native: uint8 ``[S, N, K//2]`` + e4m3 ``[S, N, K//16]`` (ModelOpt rows).
    K-major (``nvfp4_t``): int32 ``[S, K//8, N]`` + ``[S, K//16, N]`` from
    ``nvfp4_fused_moe.nvfp4_transpose_bank`` -- N contiguous, strides swapped.
    """
    if packed.dtype == torch.int32:
        (pe, pkw, pn) = packed.stride()
        (se, sblk, sn) = scale.stride()
        return packed.shape[2], packed.shape[1] * 8, packed, (pe, pn, pkw), (se, sn, sblk)
    p32 = packed.view(torch.int32)
    return packed.shape[1], packed.shape[2] * 2, p32, p32.stride(), scale.stride()


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    assert packed.dtype == torch.uint8, "LUT A/B baseline kernels read native-row banks only"
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction + arithmetic
    pair dequant (:func:`_decode_nvfp4_marlin_kernel`). Accepts either bank layout
    (:func:`_bank_geometry`): native uint8 ``[S, N, K//2]`` banks (reinterpreted as
    int32 ``[S, N, K//8]``, K%8==0 for NVFP4) or K-major ``nvfp4_t`` int32 banks."""
    M, top_k = topk_ids.shape
    N, K, packed_i32, p_strides, s_strides = _bank_geometry(packed, scale)
    kmajor = packed.dtype == torch.int32
    block_n = _DECODE_MARLIN_KM_BLOCK_N if kmajor else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_KM_BLOCK_KW if kmajor else _DECODE_MARLIN_BLOCK_KW
    warps = _DECODE_MARLIN_KM_WARPS if kmajor else _DECODE_MARLIN_WARPS
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        total_routes, N, K,
        a.stride(0), a.stride(1),
        *p_strides,
        *s_strides,
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=warps,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    # gate_up N dim: [S, N, K//2] native rows vs [S, K//8, N] K-major banks
    two_i = (gate_up_packed.shape[2] if gate_up_packed.dtype == torch.int32
             else gate_up_packed.shape[1])
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def _prefill_config(M: int, kmajor: bool = False) -> Dict[str, Any]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune. Swept offline on the RTX 3060 (sm_86,
    # the WSL verification box) over (BLOCK_M, BLOCK_N, BLOCK_KW, num_warps, num_stages).
    # Native rows: bm16/bn64 anchors M<=64, bm32/bn64 won the interim sweep, bm32/bn128
    # wins every M>=512 cell; kw32 and w8 always lose (kw32+bn128 exceeds smem).
    # K-major banks: bn128 wins everywhere (1.46-2.04x over the native-best config) --
    # N-contiguous tiles feed the dots 256-512B runs instead of 64B fragments.
    if kmajor:
        if M <= 256:
            return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_KW=16,
                        GROUP_SIZE_M=1, num_warps=4, num_stages=4)
        return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_KW=16,
                    GROUP_SIZE_M=8, num_warps=4, num_stages=4)
    if M <= 256:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KW=16,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_KW=16,
                GROUP_SIZE_M=8, num_warps=4, num_stages=4)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N, K, packed_i32, p_strides, s_strides = _bank_geometry(packed, scale)
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    even_k = K % (cfg["BLOCK_SIZE_KW"] * 8) == 0
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        *p_strides,
        *s_strides,
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        EVEN_K=even_k,
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    # gate_up N dim: [S, N, K//2] native rows vs [S, K//8, N] K-major banks
    two_i = (gate_up_packed.shape[2] if gate_up_packed.dtype == torch.int32
             else gate_up_packed.shape[1])
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M, kmajor=gate_up_packed.dtype == torch.int32)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    _run_act(activation, ic1.view(-1, two_i), ic2, act_alpha, act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
