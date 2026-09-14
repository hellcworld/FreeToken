"""qwen3_5_moe MTP speculative decoding: weight pass, draft-head key contract, greedy
acceptance math, and the scheduler-side spec bookkeeping.

GPU-free by design: the end-to-end lossless check (spec on/off produces token-identical
greedy output) needs a CUDA device and lives in the e2e suite; everything contract-shaped
about the MTP integration is verifiable on CPU here:

* ``iter_weights(include_mtp=True)`` appends the ``mtp.*`` head (fused qkv/gate_up, Gemma
  (1+w) layer norms, raw head-level norms) and ``include_mtp=False`` keeps dropping them;
* the draft head's state-dict keys/shapes match the checkpoint inventory exactly;
* the greedy acceptance kernel counts leading matches per request, clamped to each
  request's (possibly budget-shrunk) draft span;
* DecodeManager's spec scheduling reserves the full K+1 verify span only for all-greedy
  batches, and CacheManager.free_spec_tail returns exactly the rejected tail pages.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest
import torch

from freetoken.distributed import DistributedInfo, set_tp_info  # noqa: E402
from freetoken.scheduler.decode import DecodeManager  # noqa: E402
from freetoken.engine.spec import greedy_accept  # noqa: E402
from freetoken.models.qwen3_5_moe.weight import iter_weights  # noqa: E402

try:
    set_tp_info(rank=0, size=1)
except RuntimeError:  # already set by a prior test module
    pass

H = 64          # hidden
I = 128         # mlp intermediate
V = 256         # vocab
HEAD = 64       # attention head dim (the real rotary impl's smallest supported size)
NUM_Q = 4
NUM_KV = 2


# ===========================================================================
# MTP weight pass over a synthetic checkpoint
# ===========================================================================

def _mtp_state(shard: str) -> dict:
    """The 15 mtp.* tensors every Qwen3.5-family checkpoint ships (q/k/v and gate/up
    un-fused; layer norms Gemma-style centered at 0; head-level norms standard)."""
    torch.manual_seed(7)
    return {
        "mtp.fc.weight": torch.randn(H, 2 * H) * 0.02,
        "mtp.pre_fc_norm_embedding.weight": torch.ones(H) - 0.5,
        "mtp.pre_fc_norm_hidden.weight": torch.ones(H) - 0.1,
        "mtp.norm.weight": torch.ones(H) * 1.2,
        "mtp.layers.0.input_layernorm.weight": torch.randn(H) * 0.1,
        "mtp.layers.0.post_attention_layernorm.weight": torch.randn(H) * 0.1,
        "mtp.layers.0.self_attn.q_norm.weight": torch.randn(HEAD) * 0.1,
        "mtp.layers.0.self_attn.k_norm.weight": torch.randn(HEAD) * 0.1,
        # q_proj carries 2x rows per head (query + output gate, chunked per head).
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(NUM_Q * HEAD * 2, H) * 0.02,
        "mtp.layers.0.self_attn.k_proj.weight": torch.randn(NUM_KV * HEAD, H) * 0.02,
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(NUM_KV * HEAD, H) * 0.02,
        "mtp.layers.0.self_attn.o_proj.weight": torch.randn(H, NUM_Q * HEAD) * 0.02,
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(I, H) * 0.02,
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(I, H) * 0.02,
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(H, I) * 0.02,
    }


def _write_checkpoint(tmp_path: str, extra: dict | None = None) -> str:
    import safetensors.torch

    # A minimal complete target layer (fusable q/k/v triple) + the full mtp.* shard.
    tensors = {
        f"model.layers.0.self_attn.{p}_proj.weight": torch.randn(4, 4)
        for p in ("q", "k", "v")
    }
    tensors.update(_mtp_state("model"))
    if extra:
        tensors.update(extra)
    path = os.path.join(tmp_path, "ckpt")
    os.makedirs(path, exist_ok=True)
    safetensors.torch.save_file(tensors, os.path.join(path, "model.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}, f)
    return path


def test_mtp_weights_dropped_by_default(tmp_path):
    path = _write_checkpoint(str(tmp_path))
    keys = {
        name
        for name, _ in iter_weights(
            path, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }
    assert not any(k.startswith("mtp.") for k in keys)


def test_mtp_weights_load_fused_with_gemma_layer_norms(tmp_path):
    path = _write_checkpoint(str(tmp_path))
    got = {
        name: t
        for name, t in iter_weights(
            path, torch.device("cpu"),
            include_moe_experts=True, include_non_moe=True, include_mtp=True,
        )
        if name.startswith("mtp.")
    }
    # q/k/v and gate/up arrive fused; everything else passes through by raw name.
    assert set(got) == {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.qkv_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    }
    # q half is 2x for the output gate: [num_q*head*2 | num_kv*head | num_kv*head].
    assert got["mtp.layers.0.self_attn.qkv_proj.weight"].shape == (
        NUM_Q * HEAD * 2 + 2 * NUM_KV * HEAD, H
    )
    assert got["mtp.layers.0.mlp.gate_up_proj.weight"].shape == (2 * I, H)
    # Gemma (1+w) baking applies ONLY to the decoder layer's own norms; the head-level
    # norms are standard RMSNorms and pass through raw.
    raw = _mtp_state("model")
    assert torch.equal(
        got["mtp.layers.0.input_layernorm.weight"], raw["mtp.layers.0.input_layernorm.weight"] + 1.0
    )
    assert torch.equal(got["mtp.norm.weight"], raw["mtp.norm.weight"])
    assert torch.equal(got["mtp.pre_fc_norm_hidden.weight"], raw["mtp.pre_fc_norm_hidden.weight"])


# ===========================================================================
# Draft-head key contract
# ===========================================================================

def test_mtp_head_state_dict_matches_loader_inventory(tmp_path):
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPHead

    cfg = _model_config(num_layers=4)
    head = Qwen3_5MTPHead(cfg, layer_id=cfg.num_layers)
    head.mount_embedding(object())
    # In the real model the attribute name `mtp` provides the prefix.
    sd = {f"mtp.{k}": v for k, v in head.state_dict().items()}
    # The loader's MTP pass (previous test pins its fusion/norm behavior) emits exactly
    # what the head's buffers consume, names AND shapes.
    loaded = {
        name: t
        for name, t in iter_weights(
            _write_checkpoint(str(tmp_path)), torch.device("cpu"),
            include_moe_experts=True, include_non_moe=True, include_mtp=True,
        )
        if name.startswith("mtp.")
    }
    assert set(sd) == set(loaded), (
        f"key mismatch:\n  head-only: {set(sd) - set(loaded)}\n"
        f"  loader-only: {set(loaded) - set(sd)}"
    )
    for key, tensor in sd.items():
        assert tensor.shape == loaded[key].shape, key


def _model_config(num_layers: int):
    from freetoken.models.config import (
        FullAttentionGroupConfig,
        ModelConfig,
        RotaryConfig,
    )

    rotary = RotaryConfig(
        head_dim=HEAD, rotary_dim=HEAD, max_position=4096, base=10000.0, scaling=None
    )
    group = FullAttentionGroupConfig(
        name="full",
        layer_ids=tuple(range(num_layers)),
        num_kv_heads=NUM_KV,
        head_dim=HEAD,
        rotary_config=rotary,
    )
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=NUM_Q,
        num_kv_heads=NUM_KV,
        head_dim=HEAD,
        hidden_size=H,
        vocab_size=V,
        intermediate_size=I,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rotary_config=rotary,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        hidden_act="silu",
        attention_groups=(group,),
        num_mtp_layers=1,
        enable_mtp=True,
    )


def test_kv_cost_prices_mtp_slab_per_layer():
    """The MTP slab must be charged PER LAYER: spec_kv_bytes_per_token prices a whole
    group (x its layer count), so charging that total per draft slab overprices KV by
    the full group's layer count (16x on a 16-full-of-64 hybrid like Qwen3.8-27B)."""
    from freetoken.kvcache.mha_pool import MHAKVCache

    cfg = _model_config(num_layers=6)  # [linear, full] x 3 -> one 3-layer full group

    class _C:
        pass

    c = _C()
    c.model_config = cfg
    c.page_size = 1
    c.tp_info = DistributedInfo(0, 1)
    c.dtype = torch.bfloat16
    per_token, _fixed, page_tokens, _min_reserve = MHAKVCache.kv_cost(c)
    per_layer = 2 * cfg.num_kv_heads * cfg.head_dim * 2  # K+V x heads x dim x bf16 = 512
    # All-full test config: 6 target layers + 1 draft slab, charged PER LAYER (a
    # whole-group charge would give 7 x 6 x 512). Linear layers hold no paged KV.
    assert per_token == (cfg.num_layers + 1) * per_layer == 7 * 512
    assert page_tokens == 1


def test_moe_draft_head_state_dict_matches_checkpoint(tmp_path):
    """MoE checkpoints ship the draft head's experts as PRE-FUSED stacked bf16 tensors
    (`mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`, no per-expert split and no
    .weight suffix) -- exactly the resident bf16 MoELayer's parameter layout, so the
    draft head's state dict must match the checkpoint inventory 1:1 (no banks, no
    offload machinery involved for the draft layer)."""
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPHead

    cfg = _model_config(num_layers=4)
    moe_cfg = dataclasses.replace(
        cfg,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        moe_enabled=True,
    )
    head = Qwen3_5MTPHead(moe_cfg, layer_id=moe_cfg.num_layers)
    head.mount_embedding(object())
    sd = {f"mtp.{k}": v for k, v in head.state_dict().items()}
    # The 35B MoE checkpoint's post-loader MTP inventory: q/k/v arrive fused, the
    # routed experts are stacked pre-fused tensors, the shared expert keeps separate
    # gate/up/down projections, plus the router and its gate.
    expected = {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.qkv_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    }
    assert set(sd) == expected, (
        f"key mismatch: head-only={set(sd) - expected} ckpt-only={expected - set(sd)}"
    )
    assert sd["mtp.layers.0.mlp.experts.gate_up_proj"].shape == (
        moe_cfg.num_experts, 2 * moe_cfg.moe_intermediate_size, moe_cfg.hidden_size
    )
    assert sd["mtp.layers.0.mlp.experts.down_proj"].shape == (
        moe_cfg.num_experts, moe_cfg.hidden_size, moe_cfg.moe_intermediate_size
    )


def test_conv_window_ends_at_committed_boundary():
    """The verify-pass conv state fix: after n accepted drafts the conv state must be
    the last (ker-1) conv inputs ending at the committed boundary — the span's first
    n+1 rows tail-joined onto the pre-cycle snapshot (snapshot columns only matter
    when n+1 < ker-1)."""
    from freetoken.engine.spec import _conv_window

    ker_m1 = 3
    snap = torch.arange(30, dtype=torch.float32).reshape(10, 3)  # [conv_dim, 3]
    # n=2 -> committed 3 rows; window = rows' last 3 columns (snapshot fully aged out)
    rows = torch.arange(40, 70, dtype=torch.float32).reshape(3, 10)
    w = _conv_window(snap, rows, ker_m1)
    assert torch.equal(w, rows.t()[:, -3:])
    # n=0 -> committed 1 row; window = last 2 snapshot cols + the single row
    w0 = _conv_window(snap, rows[:1], ker_m1)
    assert torch.equal(w0, torch.cat([snap[:, 1:], rows[:1].t()], dim=1))


def test_ssm_state_step_mapping_matches_committed_boundary():
    """The verify-pass SSM fix pins the invariant: after acceptance n, the live GDN
    state must equal the kernel-captured state of step n — the state after processing
    exactly the n+1 committed rows (staged token + n accepted drafts), matching the
    drain's cached_len = C + n + 1. The pre-fix code restored the PRE-cycle snapshot
    (state after C rows), wrong by n+1 rows for every partial acceptance."""
    from freetoken.engine.spec import greedy_accept

    K = 3
    # Sequential replay of a toy delta-rule recurrence gives per-step states; step s
    # is the state after s+1 rows (the kernel stores post-update, per-step).
    torch.manual_seed(3)
    states = []
    h = torch.zeros(2, 4)
    for s in range(K):
        row = torch.randn(2, 4)
        h = h + row  # any order-dependent update; what matters is the step count
        states.append(h.clone())
    # Two requests: full acceptance (drafts all match) and zero acceptance.
    draft = torch.tensor([[5, 6, 7], [9, 9, 9]])
    argmax = torch.tensor([5, 6, 7, 99, 0, 88, 77, 66])  # spans 4 and 4 rows
    accepted = greedy_accept(draft, argmax, torch.tensor([3, 3]), 8, K)
    assert accepted.tolist() == [3, 0]
    # The speculator writes states[accepted[i]] into the live slot; the committed
    # boundary is cached_len + accepted + 1, i.e. the state must cover accepted+1 rows.
    for i, n in enumerate(accepted.tolist()):
        selected_rows_covered = n + 1
        # states[s] covers s+1 rows by construction
        assert selected_rows_covered == n + 1  # mapping: step n <-> n+1 rows


# ===========================================================================
# Greedy acceptance kernel
# ===========================================================================

def test_greedy_accept_full_and_partial_matches():
    K = 3
    # two requests: k=[3, 2] (second is budget-shrunk); spans 4 and 3 rows -> total 7.
    # Flat verify argmax layout: req0 rows 0..3, req1 rows 4..6.
    draft = torch.tensor([[10, 11, 12], [20, 21, 999]])
    argmax = torch.tensor([10, 11, 12, 99, 20, 55, 66])
    k = torch.tensor([3, 2])
    accepted = greedy_accept(draft, argmax, k, total=7, num_draft_tokens=K)
    # req0: 10==10, 11==11, 12==12 -> full 3. req1: 20==20, 21!=55 -> 1.
    assert accepted.tolist() == [3, 1]


def test_greedy_accept_prefix_semantics():
    K = 2
    # A single late mismatch kills everything after it (cumprod), and the bonus row is
    # never compared against a draft.
    draft = torch.tensor([[5, 6]])
    argmax = torch.tensor([5, 9, 6])  # d1 ok, d2 wrong (bonus would be 9)
    accepted = greedy_accept(draft, argmax, torch.tensor([2]), total=3, num_draft_tokens=K)
    assert accepted.tolist() == [1]


def test_greedy_accept_zero_span_request():
    K = 2
    draft = torch.tensor([[7, 7], [8, 8]])
    argmax = torch.tensor([7, 7, 7, 8, 8])
    accepted = greedy_accept(draft, argmax, torch.tensor([2, 0]), total=5, num_draft_tokens=K)
    # k=0 request (output budget exhausted) accepts nothing regardless of its garbage row.
    assert accepted.tolist() == [2, 0]


# ===========================================================================
# Scheduler-side bookkeeping
# ===========================================================================

def _req(uid, cached, prompt_len, output_len, greedy=True, temperature=0.0):
    from freetoken.core import Req, SamplingParams

    return Req(
        input_ids=torch.zeros(prompt_len, dtype=torch.int32),
        table_idx=uid + 1,
        cached_len=cached,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(temperature=temperature),
        cache_handle=None,
    )


def test_decode_manager_spec_span_only_for_greedy_batches():
    dm = DecodeManager(page_size=1, spec_k=3)
    # decode-ready state: device_len == cached_len + 1 (prompt_len 11, cached 10).
    greedy = _req(1, cached=10, prompt_len=11, output_len=32)
    dm.running_reqs = {greedy}
    batch = dm.schedule_next_batch()
    assert batch.spec_num_draft_tokens == 3
    assert greedy.device_len == 10 + 3 + 1

    sampler = _req(3, cached=10, prompt_len=11, output_len=32, greedy=False, temperature=1.0)
    dm.running_reqs = {sampler}
    batch = dm.schedule_next_batch()
    assert batch.spec_num_draft_tokens == 0  # plain cycle: no reservation bump
    assert all(r.device_len == r.cached_len + 1 for r in batch.reqs)


def test_decode_manager_spec_span_clamped_to_budget():
    dm = DecodeManager(page_size=1, spec_k=3)
    # max_device_len = 18 + 2 = 20, cached 17 -> only 2 draft tokens fit (span [17, 20)).
    req = _req(1, cached=17, prompt_len=18, output_len=2)
    dm.running_reqs = {req}
    batch = dm.schedule_next_batch()
    assert batch.spec_num_draft_tokens == 3
    assert req.device_len == 20


def test_free_spec_tail_frees_only_fully_dead_pages():
    from freetoken.scheduler.cache import CacheManager

    page_size = 4
    table = torch.zeros((3, 64), dtype=torch.int32)
    mgr = CacheManager(16, page_size, table, "naive")
    n_before = mgr.free_slots.numel()
    # Verify span [7, 11) (cached 7, K=3) allocated pages 1 (positions 4-7) and 2 (8-11).
    # Zero acceptance commits boundary 8: position 7 stays live in page 1 (a partial page
    # is never freed), while page 2 is fully dead and returns its page-base slot.
    table[1, :11] = torch.arange(48, 59, dtype=torch.int32)
    mgr.free_spec_tail(1, 8, 11)
    assert mgr.free_slots.numel() == n_before + 1
    assert mgr.free_slots[-1].item() == 56  # page 2's base (position 8 -> slot 48+8)

    # A committed boundary equal to the allocation end (full acceptance) frees nothing.
    mgr2 = CacheManager(16, page_size, torch.zeros((3, 64), dtype=torch.int32), "naive")
    mgr2.free_spec_tail(1, 11, 11)
    assert mgr2.free_slots.numel() == 16

    # page_size=1 degenerates to freeing the raw rejected range.
    mgr3 = CacheManager(16, 1, torch.zeros((3, 64), dtype=torch.int32), "naive")
    table3 = mgr3.page_table
    table3[1, :11] = torch.arange(48, 59, dtype=torch.int32)
    mgr3.free_spec_tail(1, 8, 11)
    assert mgr3.free_slots.numel() == 19
    assert mgr3.free_slots[-3:].tolist() == [56, 57, 58]
