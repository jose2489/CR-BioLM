"""
maps_checklist.py
-----------------
Live console checklist for per-species pipeline progress.
Prints [✓] / [✗] / [ ] markers to stdout and optionally to a logger.
"""

import sys
import logging

_UTF8 = getattr(sys.stdout, "encoding", "utf-8") and "utf" in (sys.stdout.encoding or "").lower()

_OK   = "[✓]" if _UTF8 else "[X]"
_FAIL = "[✗]" if _UTF8 else "[!]"
_PEND = "[ ]"
_BAR  = "═" * 46 if _UTF8 else "=" * 46


class Checklist:
    """
    Per-species console checklist.

    Usage:
        cl = Checklist("Sobralia amabilis — maps-only")
        cl.check("GBIF fetched", ok=True, detail="n=87")
        cl.check("Habitat map", ok=False, detail="DEM missing")
        cl.done()
    """

    def __init__(self, title: str, logger: logging.Logger | None = None):
        self.title   = title
        self.logger  = logger
        self._total  = 0
        self._failed = 0
        self._emit(_BAR)
        self._emit(f"  {title}")
        self._emit(_BAR)

    def check(self, label: str, ok: bool = True, detail: str = "") -> None:
        self._total += 1
        marker = _OK if ok else _FAIL
        if not ok:
            self._failed += 1
        suffix = f"  ({detail})" if detail else ""
        self._emit(f"{marker} {label:<38s}{suffix}")

    def pending(self, label: str) -> None:
        self._emit(f"{_PEND} {label}")

    def done(self) -> None:
        self._emit(_BAR)
        if self._failed == 0:
            status = f"  ✓ {self._total}/{self._total} checks passed" if _UTF8 else f"  OK {self._total}/{self._total} checks passed"
        else:
            status = f"  ✗ {self._failed} of {self._total} checks FAILED" if _UTF8 else f"  FAIL {self._failed} of {self._total} checks FAILED"
        self._emit(status)

    def _emit(self, msg: str) -> None:
        print(msg, flush=True)
        if self.logger:
            self.logger.info(msg)
