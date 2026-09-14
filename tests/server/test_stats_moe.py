"""MoE expert-cache miss window in /v1/stats (frontend side, no GPU).

StatsTracker keeps the last-known window (like kv/mamba) and build_stats surfaces it as
the ``moe`` block; the block is null until a --moe-collect-stats window arrives, and a
perfectly-hit window (miss_rate 0.0) must still be reported.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.message.frontend import UserReply
from freetoken.server.stats import StatsTracker, build_stats


def _state(tr: StatsTracker) -> SimpleNamespace:
    return SimpleNamespace(
        stats=tr,
        config=SimpleNamespace(
            page_size=1,
            served_model_name="m",
            max_seq_len=8192,
            model_config=SimpleNamespace(is_moe=True),
        ),
        instance_id="i",
        ready_at=None,
        gpus=[],
    )


def _reply(**kw) -> UserReply:
    return UserReply(uid=1, incremental_output="x", finished=False, **kw)


def test_window_reported_with_hit_rate_complement():
    tr = StatsTracker()
    tr.observe(
        _reply(
            moe_miss_rate=0.25,
            moe_active_per_layer=8.0,
            moe_missing_per_layer=2.0,
            moe_fetch_rate=1.0,
        )
    )
    assert tr.moe_miss_rate == 0.25
    doc = build_stats(_state(tr), 0, 0)
    assert doc["moe"]["miss_rate"] == 0.25
    assert doc["moe"]["hit_rate"] == 0.75
    assert doc["moe"]["active_per_layer"] == 8.0
    assert doc["moe"]["missing_per_layer"] == 2.0
    assert doc["moe"]["fetch_rate"] == 1.0


def test_null_until_first_window_and_kept_after():
    tr = StatsTracker()
    doc = build_stats(_state(tr), 0, 0)
    assert doc["moe"] is None

    # plain replies (no window stamped) neither report nor clear the last-known window
    tr.observe(_reply(moe_miss_rate=0.5, moe_active_per_layer=8.0, moe_missing_per_layer=4.0))
    tr.observe(_reply())
    assert tr.moe_miss_rate == 0.5


def test_zero_miss_window_is_reported_and_plain_offload_has_no_fetch_rate():
    tr = StatsTracker()
    tr.observe(
        _reply(
            moe_miss_rate=0.0,
            moe_active_per_layer=8.0,
            moe_missing_per_layer=0.0,
            moe_fetch_rate=-1.0,
        )
    )
    doc = build_stats(_state(tr), 0, 0)
    assert doc["moe"] is not None
    assert doc["moe"]["hit_rate"] == 1.0
    assert doc["moe"]["fetch_rate"] is None
