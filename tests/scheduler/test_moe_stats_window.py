"""--moe-collect-stats window sampling (scheduler side, no GPU).

Pins the cadence contract of Scheduler._moe_miss_window: the device-side miss counters
are read (and reset) exactly once per decode_log_interval decode batches, never on
prefill batches, and never when collection is off or the model has no offload cache.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler


class _Cache:
    def __init__(self, collect: bool = True) -> None:
        self.collect_stats = collect
        self.reads = 0
        self.resets = 0

    def decode_miss_stats(self) -> dict:
        self.reads += 1
        return {
            "miss_rate": 0.25,
            "active_per_layer": 8.0,
            "missing_per_layer": 2.0,
            "fetch_rate": 1.0,
        }

    def reset_stats(self) -> None:
        self.resets += 1


def _sched(cache, interval: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        engine=SimpleNamespace(moe_offload_cache=cache),
        config=SimpleNamespace(decode_log_interval=interval),
        _moe_stat_batches=0,
    )


_DECODE = SimpleNamespace(is_decode=True)
_PREFILL = SimpleNamespace(is_decode=False)


def test_window_sampled_once_per_interval():
    cache = _Cache()
    sched = _sched(cache, interval=4)
    results = [Scheduler._moe_miss_window(sched, _DECODE) for _ in range(8)]
    assert [r is not None for r in results] == [
        False, False, False, True, False, False, False, True,
    ]
    # each reported window read the counters once and reset them to delimit the window
    assert cache.reads == 2
    assert cache.resets == 2
    assert results[3]["miss_rate"] == 0.25


def test_prefill_batches_never_sample_or_advance_the_counter():
    cache = _Cache()
    sched = _sched(cache, interval=2)
    assert Scheduler._moe_miss_window(sched, _PREFILL) is None
    assert Scheduler._moe_miss_window(sched, _PREFILL) is None
    assert cache.reads == 0
    # decode cadence is unaffected by the interleaved prefills
    assert Scheduler._moe_miss_window(sched, _DECODE) is None
    window = Scheduler._moe_miss_window(sched, _DECODE)
    assert window is not None and window["missing_per_layer"] == 2.0


def test_disabled_or_absent_cache_is_a_noop():
    sched = _sched(_Cache(collect=False), interval=1)
    assert Scheduler._moe_miss_window(sched, _DECODE) is None

    sched = SimpleNamespace(
        engine=SimpleNamespace(moe_offload_cache=None),
        config=SimpleNamespace(decode_log_interval=1),
        _moe_stat_batches=0,
    )
    assert Scheduler._moe_miss_window(sched, _DECODE) is None
