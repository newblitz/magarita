"""Lightweight memory-usage logger for pipeline phase boundaries.

Usage::

    from src.utils.mem_monitor import log_mem
    log_mem("after loading S1")   # logs: "RSS after loading S1: 1.23 GB"

Falls back to a no-op if ``psutil`` is not installed so the rest of the
pipeline is never broken by a missing optional dependency.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import psutil as _psutil

    def _rss_gb() -> float:
        proc = _psutil.Process()
        return proc.memory_info().rss / (1024 ** 3)

    def log_mem(label: str = "") -> None:
        """Log the current process RSS to the root logger at INFO level."""
        try:
            rss = _rss_gb()
            log.info("RAM %s: %.2f GB RSS", label, rss)
        except Exception:  # pragma: no cover
            pass

except ImportError:  # psutil not installed → silent no-op

    def log_mem(label: str = "") -> None:  # type: ignore[misc]
        """No-op fallback when psutil is not available."""
        pass
