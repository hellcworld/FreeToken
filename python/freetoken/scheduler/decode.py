from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from freetoken.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    # MTP draft tokens per cycle (0 = speculation disabled). A cycle is speculative when
    # every request in it samples greedily — non-greedy requests fall back to plain
    # single-token cycles, so a mixed population simply alternates batch shapes.
    spec_k: int = 0

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        # 1 page reserved per request for the plain 1-token step's page crossing. A spec
        # cycle admits up to K+1 tokens at once, crossing up to ceil((K+1)/ps) pages —
        # cover the extra rows or admission can over-commit and the alloc assert fires.
        step_rows = max(self.spec_k + 1, 1) if self.spec_k else 1
        tokens_reserved = (
            -(-step_rows // self.page_size) * self.page_size - 1
        ) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = sorted(self.running_reqs, key=lambda req: req.uid)
        batch = Batch(reqs=reqs, phase="decode")
        if self.spec_k and all(req.sampling_params.is_greedy for req in reqs):
            batch.spec_num_draft_tokens = self.spec_k
            # Reserve the full verify span up front: _prepare_batch allocates the pages
            # for [cached_len, device_len) and the attention metadata reads these lens.
            # Cycles near the output budget shrink K instead of failing admission.
            for req in reqs:
                req.device_len = min(req.cached_len + self.spec_k + 1, req.max_device_len)
            # Snapshot for the speculator: it must see exactly these lens, not whatever
            # the previous batch's drain leaves on the Req by launch time.
            batch.spec_req_lens = [(r.cached_len, r.device_len) for r in reqs]
        return batch

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
