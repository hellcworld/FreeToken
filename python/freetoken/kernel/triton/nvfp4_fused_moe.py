"""Inline-dequant NVFP4 fused-MoE Triton kernels.

These kernels read the NVFP4 expert cache (packed FP4 codes + fp8 block scales +
fp16 per-row global scale) *directly* and dequantize inside the GEMM K-loop, so the
grouped MoE GEMM never materializes a BF16 copy of the experts (no separate dequant
pass, no HBM round-trip of the 4x larger BF16 weights).

Layout, for an expert weight ``W[N, K]``:
  - ``packed[slot, n, k//2]`` uint8: two FP4 codes per byte, low nibble = even k.
  - ``scale[slot, n, k//16]`` fp8-e4m3: per 16-wide block scale.
  - ``global[slot, n]`` fp16: per-output-row scale (``weight_scale_2``).
  - ``W[n, k] = E2M1[code] * scale[n, k//16] * global[n]``.

The global scale is constant along K, so it is applied once after the K-loop.

Decode (M=1) is HBM-bandwidth bound. :func:`_decode_nvfp4_marlin_kernel` is the production
decode GEMV (int32 wide loads + deferred K reduction + arithmetic pair dequant, no LUT
gather); :func:`_decode_nvfp4_moe_kernel` is the original LUT-gather variant kept for A/B
comparison. The wide load lifted MiniMax-M2 decode toward the RTX 5090 read-bandwidth
ceiling (~87%); the pair bit-trick removed the 8-per-word L1 LUT gathers that serialized
the LSU (the same fix the dense W4A16 kernels made earlier, where the gather alone cost
~28 points of roofline on H100). The residual gap is a swizzled-layout tensor-core path
(marlin / flashinfer b12x), which needs sm_80-99 / CUDA>=13 respectively.

Prefill runs the grouped GEMM the same way: :func:`_prefill_nvfp4_moe_kernel` loads the
packed codes as int32 words and dequants pairs to exact fp16 (bit trick + ``scale * 2^7``
folding, deferred ``g * 2^7`` in the epilogue -- the numerics proven out in
``nvfp4_linear._nvfp4_gemm_kernel``), feeding swapped-operand tensor-core dots so the
dequantized weight tile never touches shared memory. :func:`_prefill_nvfp4_moe_kernel_lut`
(byte loads + LUT gather + a strided activation load pair) is the original kernel kept
for A/B benchmarking.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import (
    e4m3_native_cx,
    e4m3_u8_to_f16_x128,
    e4m3_u8_to_f32,
)
from freetoken.kernel.triton.nvfp4_linear import (
    _nvfp4_pair_f16,
    _nvfp4_pair_f32,
    _split8,
    _split8_rows,
)

_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


@functools.lru_cache(maxsize=None)
def _e2m1_lut(device_index: int) -> torch.Tensor:
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@triton.jit
def _decode_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,  # bytes processed per K-iter (covers 2*KB k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Original LUT-gather serial-K decode GEMV. Kept as the A/B baseline; the production
    decode path is :func:`_decode_nvfp4_marlin_kernel`."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    K_BYTES = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kb_start in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb_start * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_slot + offs_n[None, :] * stride_pn + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(
            p_ptrs, mask=byte_mask[:, None] & n_mask[None, :], other=0
        ).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        b_lo = tl.load(lut_ptr + lo)
        b_hi = tl.load(lut_ptr + hi)

        sblk = byte_idx // 8
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + sblk[:, None] * stride_sblk
        s_mask = byte_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))
        b_lo = b_lo * scale
        b_hi = b_hi * scale

        a_lo = tl.load(a_base + (2 * byte_idx) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        a_hi = tl.load(a_base + (2 * byte_idx + 1) * stride_ak, mask=byte_mask, other=0.0).to(tl.float32)
        accumulator += tl.sum(a_lo[:, None] * b_lo, axis=0)
        accumulator += tl.sum(a_hi[:, None] * b_hi, axis=0)

    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _decode_nvfp4_marlin_kernel(
    a_ptr,             # [M, K] activations (compute dtype)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per K-iter (covers 8*KW k-values)
    TOP_K: tl.constexpr,
    A_ROW_IS_ROUTE: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Marlin-style NVFP4 decode GEMV: wide int32 weight loads + deferred reduction +
    arithmetic pair dequant.

    Versus :func:`_decode_nvfp4_moe_kernel` (which loads the packed codes one byte at a
    time and runs a cross-lane reduction *every* K-iter) this:

      * Loads the FP4 codes as **int32 words** (8 codes / 4 bytes per element) so the HBM
        read issues wide coalesced transactions -- the single biggest lever for the
        gate/up GEMM, whose mem-only ceiling jumps from ~65% to ~85% of peak read BW.
      * **Defers** the K reduction: the ``[BLOCK_KW, BLOCK_N]`` partial accumulates across
        *all* K-iters and is reduced once at the end, removing the per-iter ``tl.sum``
        barrier so the weight loads of successive iters pipeline.
      * Dequants **arithmetically** (:func:`_nvfp4_pair_f32`): two e2m1 codes per 5 integer
        ops on the fp16-subnormal bit trick, instead of 8 L1 LUT gathers per word. The
        gather serialized the LSU (measured ~28 roofline points on the dense twin, H100);
        the block scale is folded with the trick's ``2^-14`` into one fp32 multiply.

    Mirrors the fast kernel's route/tile mapping and epilogue so it is a drop-in decode
    GEMV (CUDA-graph safe: fixed shapes, no host sync)."""
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_row = route_id if A_ROW_IS_ROUTE else token_id
    a_base = a_ptr + a_row * stride_am

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    partial = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        # 8 codes/word fall in the same or adjacent 16-wide block -> one scale per word.
        # * 16384.0 undoes the pair trick's 2^-14 (exact, power of two).
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + (widx[:, None] // 2) * stride_sblk
        s_mask = w_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32) * 16384.0
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0)) * 16384.0

        offs_k = kw_start * BLOCK_SIZE_KW * 8 + tl.arange(0, BLOCK_SIZE_KW * 8)
        a_vec = tl.load(a_base + offs_k * stride_ak, mask=offs_k < K, other=0.0).to(tl.float32)
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8(a_vec, BLOCK_SIZE_KW)

        # Nibble j of word w is k = 8*w + j; the pair trick yields (j=0, j=4), (1, 5), ...
        acc_w = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        b_lo, b_hi = _nvfp4_pair_f32(word)
        acc_w += a0[:, None] * b_lo
        acc_w += a4[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 4)
        acc_w += a1[:, None] * b_lo
        acc_w += a5[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 8)
        acc_w += a2[:, None] * b_lo
        acc_w += a6[:, None] * b_hi
        b_lo, b_hi = _nvfp4_pair_f32(word >> 12)
        acc_w += a3[:, None] * b_lo
        acc_w += a7[:, None] * b_hi
        partial += acc_w * scale

    accumulator = tl.sum(partial, axis=0)
    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


@triton.jit
def _prefill_nvfp4_moe_kernel_lut(
    a_ptr,             # [M, K] activations
    packed_ptr,        # [S, N, K // 2] uint8
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    lut_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkb,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Original byte-load + LUT-gather grouped GEMM (fp32 dequant, two dots per byte
    tile). Kept as the A/B baseline; the production prefill kernel is
    :func:`_prefill_nvfp4_moe_kernel`."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    a_ptrs_lo = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb)[None, :] * stride_ak)
    a_ptrs_hi = a_ptr + (offs_token[:, None] // top_k * stride_am + (2 * offs_kb + 1)[None, :] * stride_ak)

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_BYTES = K // 2
    for kb in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
        byte_idx = kb * BLOCK_SIZE_KB + offs_kb
        byte_mask = byte_idx < K_BYTES

        p_ptrs = packed_base + byte_idx[:, None] * stride_pkb
        bytes_ = tl.load(p_ptrs, mask=byte_mask[:, None], other=0).to(tl.int32)
        lo = bytes_ & 0xF
        hi = (bytes_ >> 4) & 0xF
        sblk = byte_idx // 8
        s_ptrs = scale_base + sblk[:, None] * stride_sblk
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=byte_mask[:, None], other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=byte_mask[:, None], other=0))
        b_lo = tl.load(lut_ptr + lo) * scale  # [BLOCK_KB, BLOCK_N]
        b_hi = tl.load(lut_ptr + hi) * scale

        a_lo = tl.load(a_ptrs_lo, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        a_hi = tl.load(a_ptrs_hi, mask=token_mask[:, None] & byte_mask[None, :], other=0.0)
        accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
        accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

        a_ptrs_lo += BLOCK_SIZE_KB * 2 * stride_ak
        a_ptrs_hi += BLOCK_SIZE_KB * 2 * stride_ak

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32)
    accumulator = accumulator * g[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _prefill_nvfp4_moe_kernel(
    a_ptr,             # [M, K] activations (bf16/fp16)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k = 8*w + j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per K-iter (covers 8*KW k-values)
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Word-load grouped GEMM: int32 wide weight loads + arithmetic fp16 pair dequant.

    Ports the dense W4A16 GEMM's numerics and operand orientation (see
    ``nvfp4_linear._nvfp4_gemm_kernel``) onto the routed grouped-GEMM structure:

      * Packed codes load as **int32 words** (8 codes each) from a ``[S, N, K//8]``
        view, dequanted in-register by the fp16 subnormal bit trick -- no LUT gathers,
        no byte-granular transactions, no per-byte scale reloads.
      * The block scale folds as ``scale * 2^7`` into exact fp16; the deferred ``2^7``
        rides the epilogue's ``global * 2^7`` (bit-identical to scaling in-loop).
      * Swapped-operand dots (``acc^T = W_dequant @ a^T``): the dequanted weight tile is
        the lhs MMA operand, so it never takes a shared-memory round trip; the
        activation tile loads once per word tile as a contiguous ``[BM, 8*KW]`` block
        and is register-split per nibble position (:func:`_split8_rows`) instead of the
        two strided (stride-2) half loads.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[:, None] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[:, None] * stride_sn
    a_base = a_ptr + (offs_token // top_k)[:, None] * stride_am

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for kw in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw * BLOCK_SIZE_KW + offs_kw
        if EVEN_K:
            word = tl.load(packed_base + widx[None, :] * stride_pkw)
            s_ptrs = scale_base + (widx[None, :] // 2) * stride_sblk
            if e4m3_native_cx():
                s128 = tl.load(s_ptrs).to(tl.float16) * 128.0
            else:
                s128 = e4m3_u8_to_f16_x128(tl.load(s_ptrs))
            offs_k = kw * BLOCK_SIZE_KW * 8 + tl.arange(0, BLOCK_SIZE_KW * 8)
            a_tile = tl.load(a_base + offs_k[None, :] * stride_ak, mask=token_mask[:, None], other=0.0)
        else:
            w_mask = widx < K_WORDS
            word = tl.load(packed_base + widx[None, :] * stride_pkw, mask=w_mask[None, :], other=0)
            s_ptrs = scale_base + (widx[None, :] // 2) * stride_sblk
            if e4m3_native_cx():
                s128 = tl.load(s_ptrs, mask=w_mask[None, :], other=0.0).to(tl.float16) * 128.0
            else:
                s128 = e4m3_u8_to_f16_x128(tl.load(s_ptrs, mask=w_mask[None, :], other=0))
            offs_k = kw * BLOCK_SIZE_KW * 8 + tl.arange(0, BLOCK_SIZE_KW * 8)
            a_tile = tl.load(
                a_base + offs_k[None, :] * stride_ak,
                mask=token_mask[:, None] & (offs_k[None, :] < K), other=0.0,
            )

        # Nibble j of word w is k = 8*w + j; pair trick yields (j=0, j=4), (1, 5), ...
        # b carries 2^-14 and s128 carries 2^7; the epilogue restores both via g * 2^7.
        a0, a1, a2, a3, a4, a5, a6, a7 = _split8_rows(
            a_tile.to(tl.float16), BLOCK_SIZE_M, BLOCK_SIZE_KW
        )
        b_lo, b_hi = _nvfp4_pair_f16(word)
        acc = tl.dot(b_lo * s128, tl.trans(a0), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a4), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 4)
        acc = tl.dot(b_lo * s128, tl.trans(a1), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a5), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 8)
        acc = tl.dot(b_lo * s128, tl.trans(a2), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a6), acc)
        b_lo, b_hi = _nvfp4_pair_f16(word >> 12)
        acc = tl.dot(b_lo * s128, tl.trans(a3), acc)
        acc = tl.dot(b_hi * s128, tl.trans(a7), acc)

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32) * 128.0
    acc = acc * g[:, None]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        acc = acc * moe_weight[None, :]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[None, :] * stride_cm + offs_cn[:, None] * stride_cn
    c_mask = token_mask[None, :] & (offs_cn[:, None] < N)
    tl.store(c_ptrs, acc.to(compute_type), mask=c_mask)


def nvfp4_transpose_bank(
    packed: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native rows -> K-major resident layout: ``[S, N, K//2]`` uint8 +
    ``[S, N, K//16]`` e4m3 -> int32 ``[S, K//8, N]`` + ``[S, K//16, N]``.

    Same word-exact construction as ``nvfp4_linear.nvfp4_transpose_resident``, batched
    over slots: the int32 *words* move atomically (a byte-level ``.t()`` would scatter
    each word's 4 bytes across the N axis). An exact permutation -- dequantized values
    are bit-identical; only the strides the kernels see change (N contiguous), which
    unlocks wide BLOCK_N tiles (native rows made bn32 60% slower; K-major bn64/kw32
    wins every decode shape and bn128 doubles prefill on RTX 3060).
    """
    return (
        packed.view(torch.int32).permute(0, 2, 1).contiguous(),
        scale.permute(0, 2, 1).contiguous(),
    )


__all__ = [
    "_decode_nvfp4_moe_kernel",
    "_decode_nvfp4_marlin_kernel",
    "_prefill_nvfp4_moe_kernel",
    "_prefill_nvfp4_moe_kernel_lut",
    "_e2m1_lut",
    "nvfp4_transpose_bank",
]
