"""A30 — Semantic Clarity.

Detects and optionally fixes ambiguity in chunk content.
Uses DomainContext from A21 for acronym expansion and entity resolution.

Detects: ambiguous pronouns, undefined acronyms, vague entities,
unresolved references, incomplete procedures, complex sentences,
sequence gaps.

Fix modes per issue type: detect_only, rule_fix, llm_fix.
"""

from .clarity import ClarityChecker, A30Config

__all__ = ["ClarityChecker", "A30Config"]
