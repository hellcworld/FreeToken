"""CPU-native NVFP4 quantize/dequantize (reference implementations).

``dequant_nvfp4_reference`` mirrors ``kernel.triton.nvfp4_dequant`` exactly -- two
FP4 e2m1 codes per byte (low nibble = even element), one fp8-e4m3 scale per 16
elements, times a per-output-row global scale -- so it can serve as the CUDA-less
fallback of the checkpoint-conversion path (``weight = fp4 * block * global``).

``quantize_nvfp4_rtn`` is the round-trip inverse (round-to-nearest with the modelopt
global-scale recipe ``g = amax / (448 * 6)``). It is used at load time to fold
mixed-precision checkpoints whose dense MLP layers arrive in a *different* weight
format than the NVFP4 the model buffers expect (e.g. an llm-compressor export that
quantizes the trailing layers FP8 instead of FP4) into the same packed layout.
"""

from __future__ import annotations

import torch

# E2M1 (NVFP4) magnitudes indexed by the 3 magnitude bits; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = torch.finfo(torch.float8_e4m3fn).tiny


def _e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """uint8 nibble codes -> float values (sign bit 3, LUT above)."""
    lut = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)
    sign = torch.where(codes >= 8, -1.0, 1.0)
    return sign * lut[(codes % 8).to(torch.long)]


def dequant_nvfp4_reference(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``[O, IN//2] uint8 + [O, IN//16] fp8-e4m3 + per-tensor scalar -> [O, IN] dtype``.

    Bit-exact analog of the triton kernel's math (fp32 accumulation order aside),
    runnable on CPU. ``weight_scale_2`` is the modelopt per-tensor global scale;
    the kernel callers broadcast it per output row, so accept ``[1]``/``[O]``/scalar.
    """
    o, half = weight.shape
    inout = half * 2
    lo = weight & 0xF
    hi = weight >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(o, inout)  # element 2b, 2b+1
    vals = _e2m1_decode(codes)
    block = weight_scale.to(torch.float32).repeat_interleave(16, dim=-1)[:, :inout]
    g = weight_scale_2.reshape(-1)[:1].to(torch.float32)
    return (vals * block * g).to(dtype)


def _round_e2m1_idx(x: torch.Tensor) -> torch.Tensor:
    """Magnitudes (>=0) -> e2m1 magnitude-bin index 0..7, ties away from zero."""
    edges = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), dtype=torch.float32)
    return torch.bucketize(x, edges, right=False)


def quantize_nvfp4_rtn(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16/fp32 ``[O, IN]`` -> ``(packed uint8 [O, IN//2], block fp8-e4m3 [O, IN//16],
    per-tensor global fp16 [1])`` in the checkpoint-native NVFP4 layout.

    Round-to-nearest with the modelopt recipe: ``g = amax / (448 * 6)`` keeps every
    block scale ``= block_amax / (6 g)`` inside e4m3's range, matching how the
    calibrated checkpoints store their two-level scales.
    """
    w = weight.to(torch.float32)
    o, inout = w.shape
    if inout % 16:
        raise ValueError(f"NVFP4 RTN needs IN % 16 == 0, got {inout}")
    amax = w.abs().amax().clamp_min(_E4M3_MIN_NORMAL)
    g = (amax / (_E4M3_MAX * _E2M1_MAX)).to(torch.float16).to(torch.float32)
    g = g.clamp_min(torch.finfo(torch.float16).tiny)

    blocks = w.reshape(o, inout // 16, 16)
    block_amax = blocks.abs().amax(dim=-1).clamp_min(0.0)
    # e4m3-representable block scale; zero blocks get the min normal (codes round to 0).
    block_scale = (block_amax / (_E2M1_MAX * g)).clamp(
        min=float(_E4M3_MIN_NORMAL), max=_E4M3_MAX
    ).to(torch.float8_e4m3fn)
    bs = block_scale.to(torch.float32)
    q = blocks / (bs.unsqueeze(-1) * g)  # in [-6, 6] (up to e4m3 rounding)
    q = q.clamp(-_E2M1_MAX, _E2M1_MAX)

    sign = torch.where(q < 0, 8, 0)
    idx = _round_e2m1_idx(q.abs())
    codes = (idx + sign).to(torch.uint8).reshape(o, inout)

    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, block_scale.contiguous(), g.to(torch.float16).reshape(1)


__all__ = ["dequant_nvfp4_reference", "quantize_nvfp4_rtn"]
