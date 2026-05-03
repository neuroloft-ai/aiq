"""A13 — Structure.

Ensures every section has a descriptive heading that helps retrieval.
Detects headings from HTML/styles, generates headings for orphaned content,
replaces generic headings ("Misc", "Notes", "Other").

Rule-based first, LLM optional upgrade for heading generation.
"""

from .structurer import Structurer, A13Config

__all__ = ["Structurer", "A13Config"]
