"""Shared test bootstrap.

On CUDA dev boxes (Linux + the pinned triton/flashlib wheels) this is a no-op: every
real module imports and nothing is registered. On a CPU-only Windows host the freetoken
import chain still pulls ``triton`` / ``flashlib`` at module import time; stub them
BEFORE the first freetoken import so the CPU-runnable suites (weight loaders, scheduler
bookkeeping, config surfaces) stay executable here too.

transformers is imported FIRST on purpose: its backend detection resolves optional
dependencies once, before the fake triton exists, so ``from transformers import
AutoTokenizer`` keeps working downstream.
"""

from __future__ import annotations

import sys
import types

_STUBbed = getattr(types, "_freetoken_test_stubs", False)
if not _STUBbed and "freetoken" not in sys.modules:
    try:
        import triton  # noqa: F401
    except ImportError:

        class _Any:
            def __init__(self, name: str = "stub"):
                object.__setattr__(self, "__name__", name)

            def __call__(self, *a, **k):
                # Passthrough decorator when applied to a function; else an inert value.
                if len(a) == 1 and not k and hasattr(a[0], "__code__"):
                    return a[0]
                return _ANY

            def __getattr__(self, n):
                return _ANY

            def __getitem__(self, i):
                return _ANY

            def __iter__(self):
                return iter(())

            def __repr__(self):
                return "<stub>"

        _ANY = _Any()

        def _make(name: str) -> types.ModuleType:
            mod = types.ModuleType(name)
            mod.__path__ = []
            mod.__file__ = f"<stub:{name}>"
            mod.__spec__ = None
            mod.__loader__ = None
            mod.__getattr__ = lambda attr, _n=name: _get(f"{_n}.{attr}")
            return mod

        def _get(name: str) -> types.ModuleType:
            if name not in sys.modules:
                sys.modules[name] = _make(name)
            return sys.modules[name]

        import transformers  # noqa: F401

        for _base in (
            "triton", "triton.backends", "triton.backends.compiler", "triton.language",
            "triton.runtime", "triton.runtime.jit", "triton.compiler",
            "triton.compiler.compiler", "flashlib", "flashlib.kernels",
            "flashlib.kernels.slot_cache",
            # referenced via import statements (no attribute lookup involved):
            "triton.language.extra", "triton.language.extra.cuda",
        ):
            sys.modules[_base] = _make(_base)
        sys.modules["triton"].__version__ = "3.6.0"
        sys.modules["triton"].jit = lambda *a, **k: (lambda f: f)
        sys.modules["triton"].__getattr__ = lambda attr: _ANY
        sys.modules["triton.language"].__getattr__ = lambda attr: _ANY
        sys.modules["triton.runtime.jit"].__getattr__ = lambda attr: _ANY
        sys.modules["triton.language.extra.cuda"].__getattr__ = lambda attr: _ANY()
        sys.modules["flashlib.kernels.slot_cache"].N_STATS = 0
        sys.modules["flashlib.kernels.slot_cache"].Stat = _Any("Stat")
        types._freetoken_test_stubs = True

from freetoken.distributed import set_tp_info  # noqa: E402

try:  # single-process suites share the global; first writer wins
    set_tp_info(rank=0, size=1)
except RuntimeError:  # already set
    pass
