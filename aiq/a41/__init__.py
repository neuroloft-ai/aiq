"""A41 — Q&A Generator.

Generates test Q&A pairs that validate Phase 1-3 outcomes:
  - Topic coverage: can RAG find the right chunk?
  - Governance probes: did remediation work?
  - Clarity probes: are fixes applied?
  - Consistency probes: is the winner served?

Expected answers come from actual chunk content.
"""

from .generator import QAGenerator, A41Config, QAPair, QASet

__all__ = ["QAGenerator", "A41Config", "QAPair", "QASet"]
