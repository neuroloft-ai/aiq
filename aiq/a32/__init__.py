"""A32 — Consistency.

Finds contradictions between chunks. Three layers:
  1. Coverage engine: find candidate pairs worth comparing
  2. Rule-based detectors: numeric, drift, authority, process, superseded
  3. Optional LLM judge for suspicious pairs

Uses A21 DomainContext (actors, scope qualifiers) and A22 metadata (dates)
for better detection and date-aware suggestions.

User decisions per finding: select_a, select_b, keep_both (not a conflict), accept_all.
"""

from .consistency import ConsistencyChecker, A32Config

__all__ = ["ConsistencyChecker", "A32Config"]
