"""A31 — Content Governance.

Scans chunks and decides what is safe to serve, what must be blocked,
and what needs human review before reaching end users. Detects: PII,
internal notes, placeholders, editorial artifacts, metadata leaks,
vague claims, destructive actions, broken references, escalation issues.

Uses DomainContext from A21 for destructive patterns.
No hardcoded domain vocabulary.
"""

from .classifier import Classifier, A31Config

__all__ = ["Classifier", "A31Config"]
