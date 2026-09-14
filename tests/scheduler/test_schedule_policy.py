"""--schedule-policy batch picks (pure fakes, no GPU).

Pins Scheduler._policy_pick_batch's contract: prefill_first always prefers prefill when
both sides are runnable (upstream behavior), decode_first always prefers decode, and
interleave alternates one decode batch between prefill batches so a long chunked prefill
cannot stall decode for its whole prompt. One-sided runnable states fall through under
every policy, and the launched phase feeds the interleave alternation.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler

_PREFILL = SimpleNamespace(is_prefill=True)
_DECODE = SimpleNamespace(is_prefill=False)


def _sched(policy: str, *, prefill: bool = True, decode: bool = True):
    picks: list[str] = []

    def do_prefill(_budget):
        if not prefill:
            return None
        picks.append("prefill")
        return _PREFILL

    def do_decode():
        if not decode:
            return None
        picks.append("decode")
        return _DECODE

    sched = SimpleNamespace(
        config=SimpleNamespace(schedule_policy=policy),
        prefill_budget=8192,
        _last_batch_phase=None,
        prefill_manager=SimpleNamespace(runnable=prefill, schedule_next_batch=do_prefill),
        decode_manager=SimpleNamespace(runnable=decode, schedule_next_batch=do_decode),
    )
    return sched, picks


def test_prefill_first_prefers_prefill_while_both_runnable():
    sched, picks = _sched("prefill_first")
    for _ in range(4):
        batch = Scheduler._policy_pick_batch(sched)
        assert batch is _PREFILL
    assert picks == ["prefill"] * 4


def test_decode_first_prefers_decode_while_both_runnable():
    sched, picks = _sched("decode_first")
    for _ in range(4):
        batch = Scheduler._policy_pick_batch(sched)
        assert batch is _DECODE
    assert picks == ["decode"] * 4


def test_interleave_alternates_decode_between_prefill_batches():
    sched, picks = _sched("interleave")
    for _ in range(6):
        Scheduler._policy_pick_batch(sched)
    # fresh scheduler starts with a prefill, then strictly alternates
    assert picks == ["prefill", "decode", "prefill", "decode", "prefill", "decode"]


def test_one_sided_runnable_falls_through_under_every_policy():
    for policy in ("prefill_first", "decode_first", "interleave"):
        sched, picks = _sched(policy, prefill=False)
        assert Scheduler._policy_pick_batch(sched) is _DECODE
        assert picks == ["decode"]

        sched, picks = _sched(policy, decode=False)
        assert Scheduler._policy_pick_batch(sched) is _PREFILL
        assert picks == ["prefill"]


def test_interleave_bursts_prefill_while_decode_is_empty():
    # decode drained: interleave degenerates to prefill_first back-to-back chunks
    sched, picks = _sched("interleave", decode=True)
    sched.decode_manager.runnable = False
    for _ in range(3):
        assert Scheduler._policy_pick_batch(sched) is _PREFILL
    assert picks == ["prefill"] * 3
