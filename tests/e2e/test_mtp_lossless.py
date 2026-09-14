"""MTP speculative decoding end-to-end gate: LOSSLESSNESS + throughput.

The load-bearing property of speculative decoding is that greedy output is
token-identical with and without the draft chain. This gate drives the same prompts
through the production engine twice (``--speculative-algorithm MTP`` off vs on) in two
subprocesses (an engine owns the global ctx for its whole process lifetime) and asserts
exact token equality, then reports the tok/s ratio as an informational metric — this is
a gate for CORRECTNESS, not speed.

Gated behind ``needs_weights``: set ``FREETOKEN_TEST_MODEL`` to a dense Qwen3.5-family
checkpoint that ships MTP weights (e.g. Qwen3.8-27B / Qwen3.6-27B-NVFP4) and run on an
idle CUDA GPU:

    FREETOKEN_TEST_MODEL=/path/to/Qwen3.8-27B pytest tests/e2e/test_mtp_lossless.py -q

Env knobs:
  FREETOKEN_TEST_MODEL        local checkpoint dir (required; must declare
                              text_config.mtp_num_hidden_layers)
  FREETOKEN_SPEC_K            draft chain length (default 2)
  FREETOKEN_SPEC_MAX_TOKENS   per-prompt output budget (default 256)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.needs_weights

MODEL = os.environ.get("FREETOKEN_TEST_MODEL", "")
K = int(os.environ.get("FREETOKEN_SPEC_K", "2"))
MAX_TOKENS = int(os.environ.get("FREETOKEN_SPEC_MAX_TOKENS", "256"))

PROMPTS = [
    "Explain, in exactly three sentences, why greedy speculative decoding is lossless.",
    "Write a Python function that merges two sorted lists without duplicates.",
    "Count from 1 to 20, separating each number with a comma.",
    "Summarize the plot of Romeo and Juliet in four bullet points.",
]


def _require_model() -> Path:
    if not MODEL:
        pytest.skip("set FREETOKEN_TEST_MODEL to a Qwen3.5-family checkpoint with MTP weights")
    path = Path(MODEL).expanduser()
    if not path.is_dir():
        pytest.skip(f"FREETOKEN_TEST_MODEL {MODEL} is not a local directory")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return path


def _run_worker(spec: bool) -> dict:
    """One engine per process (the global ctx lives for the process lifetime)."""
    env = dict(os.environ, FREETOKEN_SPEC_WORKER="1" if spec else "0")
    proc = subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"worker (spec={spec}) failed:\n{proc.stdout}\n{proc.stderr}")
    # The worker prints a single JSON line as its last stdout line (tqdm writes to stderr).
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    assert lines, f"worker (spec={spec}) produced no JSON: {proc.stdout}"
    return json.loads(lines[-1])


def test_mtp_spec_decoding_is_lossless():
    _require_model()
    plain = _run_worker(spec=False)
    spec = _run_worker(spec=True)
    for i, (ref, got) in enumerate(zip(plain["runs"], spec["runs"])):
        assert got["token_ids"] == ref["token_ids"], (
            f"prompt {i}: speculative decoding changed the output.\n"
            f"plain tail: {ref['token_ids'][-8:]}\nspec tail: {got['token_ids'][-8:]}"
        )
    ratio = spec["tokens_per_second"] / max(plain["tokens_per_second"], 1e-6)
    print(
        f"\n[MTP spec K={K}] plain {plain['tokens_per_second']:.1f} tok/s | "
        f"spec {spec['tokens_per_second']:.1f} tok/s | ratio {ratio:.2f}x | "
        f"tokens {spec['total_tokens']}/{plain['total_tokens']}"
    )


# ---------------------------------------------------------------------------
# worker mode: `python test_mtp_lossless.py --worker` runs one engine and prints JSON
# ---------------------------------------------------------------------------

def _worker_main() -> None:
    from transformers import AutoTokenizer

    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    spec = os.environ.get("FREETOKEN_SPEC_WORKER") == "1"
    # Both workers run the Triton backend (the backend MTP is validated on) so the
    # lossless comparison is same-kernel; memory_ratio caps the KV pool on small cards
    # (a tiny checkpoint would otherwise eat all VRAM and OOM graph capture).
    kwargs = dict(
        attention_backend="triton",
        memory_ratio=float(os.environ.get("FREETOKEN_SPEC_MEM_RATIO", "0.25")),
        # Spec cycles never graph (bs x (K+1) rows is a shape the bs x 1 decode graphs
        # cannot replay); the WSL GPU stack on the validation box also hard-crashes in
        # capture lately, so both workers run eager — the lossless comparison does not
        # need graphs.
        cuda_graph_max_bs=0,
    )
    if spec:
        kwargs.update(speculative_algorithm="MTP", speculative_num_draft_tokens=K)
    llm = LLM(model_path=MODEL, max_running_req=1, **kwargs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ))
        for p in PROMPTS
    ]
    t0 = time.perf_counter()
    outs = llm.generate(
        prompts,
        # ignore_eos: a fixed output budget exercises many spec cycles (incl. rejected
        # drafts and the GDN state rollback) and makes the plain/spec comparison
        # budget-aligned regardless of where EOS would land.
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True),
    )
    elapsed = time.perf_counter() - t0
    runs = [{"token_ids": o["token_ids"]} for o in outs]
    total = sum(len(o["token_ids"]) for o in runs)
    print(
        json.dumps({
            "runs": runs,
            "total_tokens": total,
            "tokens_per_second": total / max(elapsed, 1e-6),
        })
    )
    llm.shutdown()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        raise SystemExit("run via pytest, or with --worker")
