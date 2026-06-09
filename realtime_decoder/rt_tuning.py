"""Runtime tuning hooks for hot-rank main loops.

These are deliberately small. The big wins for tail latency on Linux are
OS-level (PREEMPT_RT kernel, CPU isolation, SCHED_FIFO, mlockall, IRQ
affinity); see docs/realtime_tuning.md for the recipes. The Python side
can still help by not letting the cyclic garbage collector run inside
the hot loop, since gen-2 GC pauses are a common source of multi-ms to
100ms hiccups in any long-running numpy code.

Usage:

    from realtime_decoder import rt_tuning

    def main_loop(self):
        with rt_tuning.gc_paused_for_main_loop(self._config):
            while True:
                ...

If the config opts out (`performance.disable_gc_in_main_loop: false`),
the context manager is a no-op and the default gc behavior is kept,
which is helpful during development for catching reference cycles.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Any, Dict, Optional


def _perf(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not config:
        return {}
    return config.get('performance') or {}


@contextlib.contextmanager
def gc_paused_for_main_loop(config: Optional[Dict[str, Any]] = None):
    """Disable cyclic gc for the duration of the main loop.

    Most hot-path allocations have been eliminated by the
    pre-allocation work elsewhere in this package, so the long-tail
    risk from gc is cyclic garbage produced by error paths, logging,
    and one-off setup work. A single pre-loop ``gc.collect()`` clears
    that out; then we keep gc disabled until the loop exits and run a
    final collect on shutdown.

    Opt out by setting ``performance.disable_gc_in_main_loop: false``
    in your config. We always re-enable gc on exit so subsequent
    process-level work (final writes, etc.) is unaffected.
    """
    enabled = _perf(config).get('disable_gc_in_main_loop', True)
    if not enabled:
        yield
        return

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()
