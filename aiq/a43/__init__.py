"""A43 — Metrics.

Computes scores from A42 test results:
  - Overall: Test 1 (raw) vs Test 2 (pipeline) + delta
  - By source: topic/governance/clarity/consistency
  - Risk score: failures relative to tolerance
  - Per-module stats
"""

from .metrics import MetricsCalculator, A43Config, MetricsResult

__all__ = ["MetricsCalculator", "A43Config", "MetricsResult"]
