"""A11 — Domain Context.

Builds the context layer for the entire pipeline. Scans content,
infers domain, extracts vocabulary, and maps entity relationships.
Produces a DomainContext object consumed by ALL downstream modules.

Three modes:
  Local:  Rule-based with strict filtering (no LLM, free)
  Hybrid: Rule-based + one LLM call to validate/clean (recommended)
  API:    LLM extracts from scratch (best quality, highest cost)
"""

from .inferrer import DomainInferrer, A11Config, get_domain_defaults

__all__ = ["DomainInferrer", "A11Config", "get_domain_defaults"]
