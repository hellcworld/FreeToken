"""MTP speculative decoding: draft chain + single-pass verify + greedy acceptance.

Per decode cycle (all tokens per request move together; shapes below for bs requests):

1. draft   — K chained passes through the MTP head (bs x 1 each). Step j predicts the
   token at position ``cached_len + j + 1`` from ``[emb(t_j), h(j-1)]`` where ``t_0`` is
   the staged (already-sampled) token and ``h(-1)`` is the stashed TARGET hidden state of
   the last committed position. Each step writes its KV row into the draft head's own
   pool slab at the SAME physical slot the target verify will use, so allocation and
   rejected-draft rollback ride the existing page table untouched.
2. verify  — ONE target forward over ``[t_0, d_1 .. d_K]`` (positions
   ``[cached_len, cached_len + K]``), submitted as a decode-phase batch with
   ``extend_len = K+1``: the Triton backend routes it to the multi-query extend kernels
   (intra-block causal + paged prefix) and both lm_heads keep every row (they only
   gather on the prefill phase).
3. accept  — greedy: ``n_i`` = leading draft tokens matching the target argmax. The
   scheduler commits ``n_i + 1`` tokens (the accepted drafts plus the argmax at the last
   accepted position as the bonus/correction token) and frees the rejected KV tail.

Output is mathematically lossless: greedy decoding with speculative verification is
token-identical to plain decoding. Draft quality (and thus the acceptance rate) depends
on the MTP head only.

Overlap note: a spec cycle CANNOT overlap with the next batch's scheduling — the page
allocation, token writes and request bookkeeping of cycle N+1 depend on cycle N's
acceptance counts, which only exist after the D2H drain. The scheduler therefore drains
a spec batch before scheduling the next one (see Scheduler.overlap_loop).
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from freetoken.core import Batch


class SpecForwardOutput(NamedTuple):
    # Target argmax over ALL verify rows, flat per padded request (k_i + 1 rows each).
    # The scheduler writes row j into token_pool slot [cached_len + 1 + j], which lands
    # the accepted drafts AND the bonus/correction token at the right slots in one
    # vectorized write (rejected rows land in slots that later cycles overwrite).
    argmax_gpu: torch.Tensor
    argmax_cpu: torch.Tensor
    # [bs] accepted draft counts (host, for the drain-side bookkeeping).
    accepted_gpu: torch.Tensor
    accepted_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class _SpecVerify:
    """Scratch the GDN layers and the speculator share across one verify forward.

    The verify pass must NOT leave per-request GDN state at "processed all K+1 rows" —
    only n+1 rows get committed. The SSM (recurrent) state runs in the kernel's
    target_verify mode (live slot untouched; per-step states captured here), and the
    conv state is rebuilt from the captured raw conv-input rows plus a pre-cycle
    snapshot (needed only when the committed window is shorter than kernel-1)."""

    def __init__(self, ssm_buffer: torch.Tensor, conv_snap: torch.Tensor) -> None:
        self.ssm_buffer = ssm_buffer  # [L_gdn, bs, K+1, HV, Kd, Vd] fp32
        self.ssm_indices = torch.arange(
            ssm_buffer.shape[1], dtype=torch.int32, device=ssm_buffer.device
        )
        self.conv_snap = conv_snap  # [L_gdn, bs, conv_dim, ker-1] (pre-cycle)
        self.conv_capture: dict[int, torch.Tensor] = {}  # li -> [total, conv_dim]


def _conv_window(snap_cols: torch.Tensor, rows: torch.Tensor, ker_m1: int) -> torch.Tensor:
    """The conv state at a committed boundary = the last ``ker-1`` conv inputs ending
    there: the span's ``rows`` ([n+1, conv_dim]) tail-joined onto the pre-cycle
    snapshot columns. Pure tensor op (CPU-testable)."""
    joined = torch.cat([snap_cols, rows.t()], dim=1)
    return joined[:, -ker_m1:]


class _DraftReq:
    """Minimal per-step request view for the draft batch's metadata builders: a decode
    step at position ``cached_len + j`` over the request's full paged history."""

    __slots__ = ("table_idx", "cached_len", "device_len")

    def __init__(self, table_idx: int, cached_len: int, device_len: int) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


def greedy_accept(
    draft_tokens: torch.Tensor,
    argmax: torch.Tensor,
    k: torch.Tensor,
    total: int,
    num_draft_tokens: int,
) -> torch.Tensor:
    """Greedy acceptance counts from the verify pass, pure-tensor (CPU-testable).

    ``draft_tokens`` [bs, K] holds the drafted chain (padded columns are garbage);
    ``argmax`` [total] is the target's per-position prediction, flat in the padded-request
    layout (k_i + 1 rows per request, row j predicting position cached_len + j + 1);
    ``k`` [bs] the per-request draft count. Returns [bs] int64 accepted counts in
    [0, k_i]: the number of leading drafts equal to the target's prediction.
    """
    bs = draft_tokens.shape[0]
    device = draft_tokens.device
    k = k.to(torch.int64)
    off = torch.cumsum(k + 1, dim=0) - (k + 1)  # exclusive prefix sums
    grid = torch.arange(num_draft_tokens, device=device)
    # Row holding the target's prediction for draft (i, j); out-of-range columns clamp to
    # the row 0 (harmless — masked below by the per-request span).
    flat_idx = (off[:, None] + grid[None, :]).clamp_max(total - 1)
    tgt = argmax[flat_idx.view(-1)].view(bs, num_draft_tokens).to(draft_tokens.dtype)
    match = (draft_tokens == tgt) & (grid[None, :] < k[:, None])
    return match.cumprod(dim=1).sum(dim=1)


class MTPSpeculator:
    def __init__(self, engine, num_draft_tokens: int) -> None:
        self.engine = engine
        self.num_draft_tokens = num_draft_tokens
        model = engine.model
        assert hasattr(model, "mtp"), "enable_mtp model must build the MTP head"
        hidden = engine.config.model_config.hidden_size
        self.hidden_size = hidden
        # Persistent per-step GDN state capture (target_verify buffer): allocating
        # ~280 MB per cycle churned the allocator next to the CPU executor's pinned
        # registrations and the KV pools; one max-size buffer sliced per cycle is
        # stable-addressed and cheaper.
        pool = engine.linear_state_pool
        self._ssm_buffer = None
        if pool is not None:
            rec = pool.recurrent_states
            L, _, HV, Kd, Vd = rec.shape
            self._ssm_buffer = torch.zeros(
                (L, engine.config.max_running_req, num_draft_tokens + 1, HV, Kd, Vd),
                dtype=rec.dtype, device=engine.device,
            )
        # Per-request stash of the target's last-position hidden state, keyed by
        # table_idx (row max_running_req = the dummy row). Draft step 0 pairs the staged
        # token with it; the verify pass re-writes it from the accepted position.
        self._stash = torch.zeros(
            (engine.config.max_running_req + 1, hidden),
            dtype=engine.dtype,
            device=engine.device,
        )

    # ------------------------------------------------------------------ prefill hook

    def note_prefill(self, batch: Batch) -> None:
        """Runs INSIDE the forward ctx right after the target's prefill forward: replays
        the chunk through the draft head KV-only (filling its slab in lockstep with the
        target's chunked prefill) and stashes each request's last-position target hidden
        for the first draft step of the coming decode cycle."""
        model = self.engine.model
        hidden = model._last_hidden
        assert hidden is not None and hidden.shape[0] == batch.input_ids.shape[0]
        metadata = batch.attn_metadata
        starts = metadata.cu_seqlens_q_gpu[: len(batch.padded_reqs)]
        # Shifted hidden pairing: token at flat row r pairs with the target hidden of the
        # previous position. Row r-1 works within a chunk; each chunk's first row pairs
        # with the stashed cross-chunk hidden (zeros for a sequence start — the stash row
        # is zeroed until the request's first chunk writes it; a stale row after a
        # table_idx reuse only costs draft quality on that request's first cycle, never
        # correctness).
        shifted = torch.cat(
            [torch.zeros((1, self.hidden_size), dtype=hidden.dtype, device=hidden.device),
             hidden[:-1]]
        )
        # Each request's first chunk row pairs with the cross-chunk hidden stash — except
        # a sequence start (cached_len == 0), whose pair is genuinely zeros. A reused
        # table_idx carrying a stale stash row only costs draft quality on that request's
        # first cycle (row 0 of a first chunk keeps zeros below regardless), never
        # correctness — the target verify absorbs any draft.
        table_rows = torch.tensor(
            [r.table_idx for r in batch.padded_reqs], dtype=torch.int64, device=hidden.device
        )
        cont = torch.tensor(
            [r.cached_len > 0 for r in batch.padded_reqs], dtype=torch.bool, device=hidden.device
        )
        starts_l = starts.long()
        # Sequence starts pair with zeros (no previous position exists); the global shift
        # left the previous request's last hidden there — zero it explicitly.
        shifted[starts_l[~cont]] = 0
        shifted[starts_l[cont]] = self._stash[table_rows[cont]]
        model.mtp.prefill(batch.input_ids, shifted)
        # Stash the last position's target hidden per request.
        last = metadata.get_last_indices(len(batch.padded_reqs))
        self._stash.index_copy_(0, table_rows, hidden[last.long()])

    def note_decode(self, batch: Batch) -> None:
        """Plain (non-spec) decode cycle: refresh the hidden stash from the forward."""
        hidden = self.engine.model._last_hidden
        table_rows = torch.tensor(
            [r.table_idx for r in batch.padded_reqs[: batch.size]],
            dtype=torch.int64,
            device=hidden.device,
        )
        self._stash.index_copy_(0, table_rows, hidden[: batch.size])

    # ------------------------------------------------------------- draft + verify

    def forward_batch(self, batch: Batch) -> SpecForwardOutput:
        engine = self.engine
        model = engine.model
        ctx = engine.ctx
        reqs = batch.reqs
        assert batch.padded_reqs == batch.reqs, "spec batches never graph-pad"
        bs = len(reqs)
        # Schedule-time spans: the overlap loop may drain the previous batch (advancing
        # the live Req lens) between this batch's scheduling and its launch, so every
        # position computation below rides the snapshot instead of the live lens.
        lens = batch.spec_req_lens or [(r.cached_len, r.device_len) for r in reqs]
        k = [max(d - c - 1, 0) for c, d in lens]  # drafts per request
        spans = [ki + 1 for ki in k]
        offsets = [0] * bs
        total = 0
        for i, s in enumerate(spans):
            offsets[i] = total
            total += s
        device = engine.device

        staged = batch.input_ids  # [bs] the already-sampled tokens (gathered by the scheduler)
        draft_tokens = torch.zeros((bs, self.num_draft_tokens), dtype=torch.int32, device=device)
        out_loc_flat = batch.out_loc

        tables = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, device=device)
        prev_hidden = self._stash.index_select(0, tables)
        cur_tokens = staged
        for j in range(self.num_draft_tokens):
            active = [i for i in range(bs) if k[i] > j]
            if not active:
                break
            rows = torch.tensor(active, dtype=torch.int64, device=device)
            d_reqs = [
                _DraftReq(reqs[i].table_idx, lens[i][0], lens[i][0] + j + 1)
                for i in active
            ]
            d_batch = Batch(reqs=d_reqs, phase="decode")  # type: ignore[arg-type]
            d_batch.padded_reqs = d_reqs  # type: ignore[assignment]
            d_batch.positions = torch.tensor(
                [lens[i][0] + j for i in active], dtype=torch.int32, device=device
            )
            flat = torch.tensor(
                [offsets[i] + j for i in active], dtype=torch.int64, device=device
            )
            d_batch.out_loc = out_loc_flat.index_select(0, flat)
            ctx.attn_backend.prepare_metadata(d_batch)
            with ctx.forward_batch(d_batch):
                hidden = model.mtp.draft(
                    cur_tokens.index_select(0, rows), prev_hidden.index_select(0, rows)
                )
                logits = model.lm_head.forward(hidden)
            d = logits.argmax(-1).to(torch.int32)
            draft_tokens[rows, j] = d
            cur_tokens = cur_tokens.index_copy(0, rows, d)
            prev_hidden = prev_hidden.index_copy(0, rows, hidden)

        # ---- verify: one target forward over [staged, d_1 .. d_k_i] per request ----
        id_parts = []
        for i in range(bs):
            id_parts.append(
                torch.cat([staged[i:i+1], draft_tokens[i, : k[i]]])
            )
        verify_ids = torch.cat(id_parts)
        batch.input_ids = verify_ids

        # GDN-hybrid models: the verify pass must leave each request's recurrent state
        # exactly at the committed boundary. The kernel runs in target_verify mode
        # (live slot untouched, per-step states captured); the conv state is rebuilt
        # post-acceptance from the captured conv-input rows. The draft head has no GDN
        # layers, so drafting never touches the pool.
        pool = engine.linear_state_pool
        sv = None
        if pool is not None and max(spans) > 1:
            # Guard max(spans) > 1: an all-span-1 batch (every request's output budget
            # clamped K to 0) takes the GDN single-step decode branch, which already
            # writes the live state correctly -- scattering the never-written buffer
            # then would zero it.
            gdn_slots = torch.tensor(
                [
                    r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx
                    for r in reqs
                ],
                dtype=torch.int64,
                device=device,
            )
            rec = pool.recurrent_states
            sv = _SpecVerify(
                ssm_buffer=self._ssm_buffer[:, :bs],
                conv_snap=pool.conv_states[:, gdn_slots].clone(),
            )
            ctx._spec_verify = sv

        with ctx.forward_batch(batch):
            hidden = model.model.forward(verify_ids)
            model._last_hidden = hidden
            logits = model.lm_head.forward(hidden)
            argmax = logits.argmax(-1).to(torch.int32)

        # ---- greedy acceptance ----
        accepted = greedy_accept(
            draft_tokens, argmax, torch.tensor(k, device=device), total, self.num_draft_tokens
        ).to(torch.int32)  # [bs] in [0, k_i]

        if pool is not None:
            ctx._spec_verify = None
            # SSM: step n's captured state == state after the n+1 committed rows; write
            # it into every GDN layer's live slot (one gather + one index_copy).
            sel = sv.ssm_buffer[:, torch.arange(bs, device=device), accepted.long()]
            pool.recurrent_states.index_copy_(1, gdn_slots, sel)
            # Conv: rebuild each layer's window ending at the committed boundary.
            ker_m1 = sv.conv_snap.shape[-1]
            accepted_host = accepted.tolist()
            for li, conv_in in sv.conv_capture.items():
                for i, req in enumerate(reqs):
                    n = accepted_host[i]
                    rows = conv_in[offsets[i] : offsets[i] + n + 1]
                    pool.conv_states[li, gdn_slots[i]] = _conv_window(
                        sv.conv_snap[li, i], rows, ker_m1
                    )

        argmax_cpu = argmax.to("cpu", non_blocking=True)
        accepted_cpu = accepted.to("cpu", non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(engine.stream)

        # Stash the target hidden at each request's last ACCEPTED position: draft step 0
        # of the next cycle pairs it with the bonus token.
        off = torch.tensor(offsets, dtype=torch.int64, device=device)
        self._stash.index_copy_(
            0,
            torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, device=device),
            hidden.index_select(0, off + accepted.long()),
        )
        return SpecForwardOutput(argmax, argmax_cpu, accepted, accepted_cpu, copy_done)
