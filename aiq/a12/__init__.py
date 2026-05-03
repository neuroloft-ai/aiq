"""A12 — Normalize.

Converts non-text content (tables, figures, procedures) into searchable text.
For each element generates BOTH:
  - Summary: topic anchor for retrieval ("This table shows payment processing times")
  - Full content: actual answer text ("Credit Card is instant at 2.9% fee...")

Keeps source references for traceability.
Rule-based for tables and procedures. LLM optional for figure descriptions.
"""

from .normalizer import Normalizer, A12Config

__all__ = ["Normalizer", "A12Config"]
