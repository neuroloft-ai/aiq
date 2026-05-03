"""A14 — Smart Chunker.

Combines topic-aware splitting with size guardrails in one pass.
Replaces old M1.5 (sizing) + M9 (coherence) as separate modules.

Algorithm:
  1. For each section, detect topic boundaries (keyword overlap between sentences)
  2. Split at topic boundaries
  3. If any piece exceeds max_words, split at sentence boundary
  4. If any piece is below min_words and shares topic with neighbor, merge
  5. Never merge chunks with different topics
  6. Assign heading from parent section (with -1, -2 suffix for splits)
"""

from .chunker import SmartChunker, A14Config

__all__ = ["SmartChunker", "A14Config"]
