"""A42 — Retrieval Test.

Runs each Q&A pair against two chunk pools:
  Test 1: A10 raw chunks (before pipeline)
  Test 2: End-of-Phase-3 chunks (after pipeline)

Judges each retrieval by comparing retrieved content to expected answer
using cosine similarity and/or LLM judge. No chunk ID matching.

Provides verdict (correct/incorrect/partial) + reasoning for each.
"""

from .retrieval import RetrievalTester, A42Config, TestResult, PairResult

__all__ = ["RetrievalTester", "A42Config", "TestResult", "PairResult"]
