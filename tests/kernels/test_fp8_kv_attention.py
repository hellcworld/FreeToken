"""fp8 paged KV: store/dequant numerics through the real pool and Triton kernels.

GPU-gated (the store path is the CUDA JIT row copier and the readers are the
Triton attention kernels). Two-level numerics contract:

1. the fp8 kernels agree with a plain fp32 attention over the DEQUANTIZED fp8
   rows -- pins the in-kernel decode/scale indexing (tight tolerance; the only
   differences are accumulation order);
2. the fp8 result stays close to the same attention over the original bf16 rows
   -- pins the accuracy claim (the quantization error survives softmax/averaging
   attenuated, well inside the 2^-4 per-element format bound).
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kvcache.mha_pool import (
    MHAKVCache,
    dequantize_rows_e4m3,
    quantize_rows_e4m3,
)
from freetoken.kernel.triton.attention import (
    decode_paged_attention,
    extend_paged_attention,
    paged_attention,
)

DEV = torch.device("cuda")

NUM_LAYERS = 2
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
HEAD_DIM = 64
NUM_SLOTS = 64
MAX_KV_SPLITS = 8
SM_SCALE = HEAD_DIM**-0.5


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(dtype):
    return MHAKVCache(
        num_kv_heads=NUM_KV_HEADS,
        num_layers=NUM_LAYERS,
        head_dim=HEAD_DIM,
        num_pages=NUM_SLOTS,
        page_size=1,
        dtype=dtype,
        device=DEV,
    )


def _ref_attention(q, k_rows, v_rows, kv_indices, indptr):
    """fp32 attention per request; q (b, hq, d), k/v rows (slots, hkv, d) already
    dequantized to fp32; ``kv_indices``/``indptr`` are the same concatenated
    per-request slot lists the kernels consume."""
    hq, hkv = q.shape[1], k_rows.shape[1]
    group = hq // hkv
    outs = []
    for b in range(q.shape[0]):
        rows = kv_indices[indptr[b]:indptr[b + 1]]
        kb = k_rows[rows].float()  # (t, hkv, d)
        vb = v_rows[rows].float()
        qb = q[b].float().view(hkv, group, -1)
        scores = torch.einsum("hgd,thd->hgt", qb, kb) * SM_SCALE
        p = scores.softmax(dim=-1)
        outs.append(torch.einsum("hgt,thd->hgd", p, vb).reshape(hq, -1))
    return torch.stack(outs)


def _ref_extend(q, kf, vf, prefix):
    """fp32 causal reference for the extend path: query token i sees the whole
    prefix plus extend rows 0..i. q (q_len, hq, d); kf/vf (kv_len, hkv, d) fp32."""
    outs = []
    for i in range(q.shape[0]):
        n = prefix + i + 1
        qb = q[i].float().view(NUM_KV_HEADS, -1, HEAD_DIM)
        kb, vb = kf[:n], vf[:n]
        s = torch.einsum("hgd,thd->hgt", qb, kb) * SM_SCALE
        p = s.softmax(dim=-1)
        outs.append(torch.einsum("hgt,thd->hgd", p, vb).reshape(NUM_Q_HEADS, -1))
    return torch.stack(outs)


def _store_both(k, v, locs, layer_id):
    """Store the same rows into a bf16 pool and an fp8 pool; return the pools plus
    the fp8 slab dequantized back to fp32 rows."""
    torch.manual_seed(7)
    k = k.contiguous().view(k.shape[0], -1)
    v = v.contiguous().view(v.shape[0], -1)
    pool16 = _pool(torch.bfloat16)
    pool8 = _pool(torch.float8_e4m3fn)
    pool16.store_kv(k, v, locs, layer_id)
    pool8.store_kv(k, v, locs, layer_id)
    k3 = k.view(-1, NUM_KV_HEADS, HEAD_DIM)
    # the stored rows must be the RNE e4m3 quantization of the inputs, byte-for-byte
    q_ref, s_ref = quantize_rows_e4m3(k3)
    stored_q = pool8.k_cache(layer_id)[locs].view(-1, NUM_KV_HEADS, HEAD_DIM)
    assert torch.equal(stored_q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.allclose(pool8.k_scale(layer_id)[locs], s_ref, rtol=0, atol=0)
    k8 = dequantize_rows_e4m3(
        pool8.k_cache(layer_id)[locs].view(-1, NUM_KV_HEADS, HEAD_DIM),
        pool8.k_scale(layer_id)[locs],
    )
    v8 = dequantize_rows_e4m3(
        pool8.v_cache(layer_id)[locs].view(-1, NUM_KV_HEADS, HEAD_DIM),
        pool8.v_scale(layer_id)[locs],
    )
    return pool16, pool8, k8, v8


def test_decode_fp8_matches_dequantized_and_stays_near_bf16():
    torch.manual_seed(0)
    batch, kv_len = 3, 16
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    k = (
        torch.randn(batch * kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV) * 2
    )
    v = torch.randn(batch * kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    locs = torch.arange(batch * kv_len, device=DEV)
    q_positions = torch.full((batch,), kv_len - 1, device=DEV)

    pool16, pool8, k8, v8 = _store_both(k, v, locs, 0)
    kc16 = pool16.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc16 = pool16.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    kc8 = pool8.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc8 = pool8.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)

    # contiguous per-request index lists (indptr over the concatenated rows)
    flat_indices = torch.arange(batch * kv_len, device=DEV)
    indptr_flat = torch.arange(batch + 1, device=DEV) * kv_len

    def run(kc, vc, ks, vs):
        logits = torch.empty(
            (batch, NUM_Q_HEADS, MAX_KV_SPLITS, HEAD_DIM),
            dtype=torch.float32, device=DEV,
        )
        lse = torch.empty(
            (batch, NUM_Q_HEADS, MAX_KV_SPLITS), dtype=torch.float32, device=DEV
        )
        splits = torch.full((batch,), MAX_KV_SPLITS, dtype=torch.int32, device=DEV)
        return decode_paged_attention(
            q=q, k_cache=kc, v_cache=vc, indptr=indptr_flat, indices=flat_indices,
            q_positions=q_positions, attn_logits=logits, attn_lse=lse,
            num_kv_splits=splits, max_kv_splits=MAX_KV_SPLITS, sm_scale=SM_SCALE,
            k_scale=ks, v_scale=vs,
        )

    out16 = run(kc16, vc16, None, None)
    out8 = run(kc8, vc8, pool8.k_scale(0), pool8.v_scale(0))

    ref16 = _ref_attention(q, kc16.float(), vc16.float(), flat_indices, indptr_flat)
    ref8 = _ref_attention(q, k8, v8, flat_indices, indptr_flat)

    # (1) kernel math over dequantized rows is faithful to plain attention
    torch.testing.assert_close(out8.float(), ref8, rtol=2e-2, atol=2e-2)
    # (2) fp8 cache stays near the bf16 cache result (quantization only)
    torch.testing.assert_close(out8.float(), ref16, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(out16.float(), ref16, rtol=2e-2, atol=2e-2)


def test_extend_fp8_prefix_matches_dequantized():
    torch.manual_seed(1)
    kv_len, q_len = 24, 8
    prefix = 16
    q = torch.randn(1, q_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)[0]
    k = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV) * 2
    v = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    locs = torch.arange(kv_len, device=DEV)
    pool16, pool8, k8, v8 = _store_both(k, v, locs, 0)

    qo_indptr = torch.tensor([0, q_len], device=DEV)
    # kv layout: the full kv of the request (prefix rows then the extend rows)
    kv_indptr = torch.tensor([0, kv_len], device=DEV)
    kv_indices = torch.arange(kv_len, device=DEV)
    prefix_lens = torch.tensor([prefix], device=DEV)
    k_ext = k.view(kv_len, NUM_KV_HEADS, HEAD_DIM)[prefix:]
    v_ext = v.view(kv_len, NUM_KV_HEADS, HEAD_DIM)[prefix:]

    def run(kc, vc, ks, vs):
        return extend_paged_attention(
            q=q, k_cache=kc, v_cache=vc, qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, prefix_lens=prefix_lens, max_q_len=q_len,
            sm_scale=SM_SCALE, k_extend=k_ext, v_extend=v_ext, k_scale=ks, v_scale=vs,
        )

    kc16 = pool16.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc16 = pool16.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    kc8 = pool8.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc8 = pool8.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)

    out16 = run(kc16, vc16, None, None)
    out8 = run(kc8, vc8, pool8.k_scale(0), pool8.v_scale(0))

    # reference: prefix from the (dequantized) cache, extend rows raw, causal
    # within the extend part (query token i sees rows 0..prefix+i)
    kfull8 = torch.cat([k8[:prefix], k_ext.float()], dim=0)
    vfull8 = torch.cat([v8[:prefix], v_ext.float()], dim=0)
    ref16 = _ref_extend(q, kc16.float(), vc16.float(), prefix)
    ref8 = _ref_extend(q, kfull8, vfull8, prefix)
    torch.testing.assert_close(out8.float(), ref8, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out8.float(), ref16, rtol=1e-1, atol=1e-1)


def test_fallback_paged_attention_fp8():
    torch.manual_seed(2)
    # 4 requests x 1 query token; request b sees rows [3b, 3b+3). One query token
    # per request at the last local position (causal window covers every row).
    batch, req_kv = 4, 3
    kv_len = batch * req_kv
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV) * 2
    v = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    locs = torch.arange(kv_len, device=DEV)
    pool16, pool8, k8, v8 = _store_both(k, v, locs, 0)

    indptr = torch.arange(batch + 1, device=DEV) * req_kv
    indices = torch.arange(kv_len, device=DEV)
    q_to_req = torch.arange(batch, device=DEV)
    q_positions = torch.full((batch,), req_kv - 1, device=DEV)

    def run(kc, vc, ks, vs):
        return paged_attention(
            q=q, k_cache=kc, v_cache=vc, indptr=indptr, indices=indices,
            q_to_req=q_to_req, q_positions=q_positions, sm_scale=SM_SCALE,
            k_scale=ks, v_scale=vs,
        )

    kc16 = pool16.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc16 = pool16.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    kc8 = pool8.k_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)
    vc8 = pool8.v_cache(0).view(-1, NUM_KV_HEADS, HEAD_DIM)

    out16 = run(kc16, vc16, None, None)
    out8 = run(kc8, vc8, pool8.k_scale(0), pool8.v_scale(0))
    ref16 = _ref_attention(q, kc16.float(), vc16.float(), indices, indptr)
    ref8 = _ref_attention(q, k8, v8, indices, indptr)
    torch.testing.assert_close(out8.float(), ref8, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out8.float(), ref16, rtol=1e-1, atol=1e-1)
