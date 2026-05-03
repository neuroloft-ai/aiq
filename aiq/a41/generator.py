"""A41 — Q&A Generator

Generates test Q&A pairs that validate Phase 1-3 pipeline outcomes.
Each question source tests a specific quality dimension.

What it generates (4 question sources):
    1. Topic coverage — one question per content chunk. Tests: can RAG find
       the right chunk? Validates retrieval accuracy.
    2. Governance probes — questions targeting remediated chunks and review-tagged
       chunks. Tests: is the safe version served? Is caveat present?
    3. Clarity probes — questions for chunks where A30 fixed issues (acronyms,
       pronouns). Tests: are clarity fixes reflected in retrieved content?
    4. Consistency probes — questions for contradictions. Tests: is the user's
       chosen winner served? Are both perspectives shown for unresolved conflicts?

How it works:
    - Recommended question counts auto-calculated from Phase 3 results
    - Fixed sources (governance, clarity, consistency) get full count
    - Topic gets remaining budget
    - Expected answers extracted from actual chunk content, not LLM-generated
    - Each pair has expected_behavior: answer / answer_safe / block / caveat
    - With LLM: natural question phrasing, targeted answers
    - Without LLM: fallback to heading-based questions, first sentences as answers

Config:
    total_questions: int (default: 0 = use recommended)
    llm_call: optional callable(prompt: str) -> str
    domain_type: str (from DomainContext, passed internally)
    include_topic/governance/clarity/consistency: bool (all True by default)

Config exposed to AIQConfig:
    eval_total_questions    -> A41Config.total_questions  (default: 0 = auto)
    user_qa_pairs           -> A41Config.user_qa_pairs   (default: [])
    eval_include_topic      -> A41Config.include_topic   (default: True)
    eval_include_governance -> A41Config.include_governance (default: True)
    eval_include_clarity    -> A41Config.include_clarity  (default: True)
    eval_include_consistency-> A41Config.include_consistency (default: True)
    (llm_call and domain_type wired from AIQConfig/A11 output)

User-provided Q&A (via user_qa_pairs):
    Users can provide their own test questions merged with auto-generated ones.
    Each: {"question": str, "expected_answer": str, "expected_behavior": "answer"|"block"|"caveat"}
    User pairs get source="user" and are never auto-deleted during dedup.
    Use expected_behavior="block" to test that unsafe content is NOT served.

Auto-detected (no user input needed):
    Recommended question counts per source — based on chunk tags and findings
    Expected answers — extracted from chunk content
    Expected behavior — determined from chunk tag and user decisions

    Users can also edit/delete generated Q&A pairs in the review UI before testing.

Input:  list[Chunk], optional A30/A31/A32 findings, optional DomainContext
Output: ModuleOutput with .data["qa_set"] = QASet containing list[QAPair]

LLM required: No (generates fallback questions from headings).
    LLM enhances: natural question phrasing, targeted expected answers.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import Chunk, DomainContext, ModuleOutput


# =====================================================================
# Types
# =====================================================================

@dataclass
class QAPair:
    """One Q&A pair for retrieval testing."""
    pair_id: str
    question: str
    expected_answer: str         # from chunk content (extracted, not generated)
    expected_behavior: str       # answer | answer_safe | block | caveat
    source_chunk_id: str = ""
    source_chunk_heading: str = ""
    source: str = "topic"        # topic | governance | clarity | consistency | user
    reasoning: str = ""
    confidence: str = "high"
    must_review: bool = False


@dataclass
class QASet:
    """Full set of Q&A pairs."""
    pairs: list = field(default_factory=list)
    total_chunks: int = 0
    chunks_covered: int = 0
    by_source: dict = field(default_factory=dict)
    by_behavior: dict = field(default_factory=dict)
    recommended_total: int = 0


# =====================================================================
# Config
# =====================================================================

@dataclass
class A41Config:
    """Configuration for Q&A generation."""
    total_questions: int = 0     # 0 = use recommended
    llm_call: Optional[Callable] = None
    domain_type: str = ""
    # Which sources to include
    include_topic: bool = True
    include_governance: bool = True
    include_clarity: bool = True
    include_consistency: bool = True
    # User-provided Q&A pairs — merged with auto-generated
    # Each: {"question": str, "expected_answer": str, "expected_behavior": str}
    user_qa_pairs: list = field(default_factory=list)


# =====================================================================
# Helpers
# =====================================================================

def _answer_from_content(content: str, max_sents: int = 3) -> str:
    """Extract first 2-3 meaningful sentences from chunk content (fallback)."""
    if not content:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', content.strip())
    good = [s.strip() for s in sents if s.strip() and len(s.strip()) > 15]
    if not good:
        return content[:200]
    return " ".join(good[:max_sents])


def _generate_expected_answer(question: str, content: str,
                              llm_call: Optional[Callable]) -> str:
    """Generate the correct answer to the question from chunk content."""
    if not llm_call:
        return _answer_from_content(content)

    try:
        prompt = (
            "Given this question and the source content, write the correct answer.\n"
            "Use ONLY information from the content. Do not add anything.\n"
            "Be concise and direct — answer the question, nothing more.\n\n"
            f"Question: {question}\n\n"
            f"Content:\n{content[:600]}\n\n"
            "Answer:"
        )
        result = llm_call(prompt)
        if result and result.strip():
            return result.strip()
    except Exception:
        pass

    return _answer_from_content(content)


def _recommend_counts(chunks: list[Chunk],
                      a30_findings: list, a31_findings: list,
                      a32_findings: list) -> dict:
    """Calculate recommended question counts per source."""
    content_chunks = [c for c in chunks if c.tag.value == "content"]
    remediated = [c for c in chunks if "remediated" in (c.tag_module or "")]
    review_chunks = [c for c in chunks if c.tag.default_behavior == "review"]

    # Chunks with A30 fixes
    fixed_chunk_ids = set()
    for f in (a30_findings or []):
        if f.fixed:
            fixed_chunk_ids.add(f.chunk_id)

    counts = {
        "topic": len(content_chunks),
        "governance": len(remediated) + len(review_chunks),
        "clarity": len(fixed_chunk_ids),
        "consistency": len(a32_findings or []),
    }
    counts["total"] = sum(counts.values())
    return counts


# =====================================================================
# Question generation — one prompt per source type
# =====================================================================

def _generate_topic_questions(chunks: list[Chunk], llm_call: Optional[Callable],
                              domain_type: str, max_count: int) -> list[QAPair]:
    """Source 1: one question per content chunk."""
    content_chunks = [c for c in chunks if c.tag.value == "content"]
    pairs = []

    for chunk in content_chunks[:max_count]:
        question = ""
        if llm_call:
            try:
                prompt = (
                    f"You are generating a test question for a {domain_type or 'general'} "
                    f"customer support knowledge base.\n\n"
                    f"CHUNK: {chunk.heading}\n"
                    f"CONTENT: {chunk.content[:400]}\n\n"
                    f"Write ONE natural question a real customer would ask that this chunk "
                    f"answers. Be specific to this content, not generic.\n\n"
                    f"Return ONLY the question."
                )
                question = llm_call(prompt)
                if question:
                    question = question.strip().split("\n")[0].strip()
            except Exception:
                pass

        if not question:
            question = f"Can you help me with {chunk.heading}?"

        pairs.append(QAPair(
            pair_id=f"qa_topic_{len(pairs) + 1}",
            question=question,
            expected_answer=_generate_expected_answer(question, chunk.content, llm_call),
            expected_behavior="answer",
            source_chunk_id=chunk.chunk_id,
            source_chunk_heading=chunk.heading,
            source="topic",
            reasoning=f"Topic coverage: tests retrieval for '{chunk.heading}'",
            confidence="high",
        ))

    return pairs


def _generate_governance_probes(chunks: list[Chunk], a31_findings: list,
                                llm_call: Optional[Callable],
                                domain_type: str, max_count: int) -> list[QAPair]:
    """Source 2: probe remediated and review-tagged chunks."""
    pairs = []

    # Remediated chunks — test if safe version is served
    remediated = [c for c in chunks if "remediated" in (c.tag_module or "")]
    for chunk in remediated[:max_count]:
        # Find original findings for this chunk
        original_findings = [f for f in (a31_findings or []) if f.chunk_id == chunk.chunk_id]
        original_reason = original_findings[0].reason if original_findings else "governance issue"

        question = ""
        if llm_call:
            try:
                prompt = (
                    f"This chunk in a {domain_type or 'general'} knowledge base was originally "
                    f"flagged for a governance issue. The sensitive content has been remediated.\n\n"
                    f"ORIGINAL ISSUE: {original_reason}\n"
                    f"CURRENT CONTENT (safe version): {chunk.content[:400]}\n\n"
                    f"Write ONE question a customer might naturally ask that would have triggered "
                    f"the original sensitive content. The correct answer should now be the "
                    f"safe/redacted version.\n\n"
                    f"Return ONLY the question."
                )
                question = llm_call(prompt)
                if question:
                    question = question.strip().split("\n")[0].strip()
            except Exception:
                pass

        if not question:
            question = f"What details are available about {chunk.heading}?"

        pairs.append(QAPair(
            pair_id=f"qa_gov_{len(pairs) + 1}",
            question=question,
            expected_answer=_generate_expected_answer(question, chunk.content, llm_call),
            expected_behavior="answer_safe",
            source_chunk_id=chunk.chunk_id,
            source_chunk_heading=chunk.heading,
            source="governance",
            reasoning=f"Governance probe: originally had {original_reason}. "
                      f"Tests if safe version is served.",
            confidence="high",
            must_review=False,  # high confidence — auto-approved
        ))

    # Review-tagged chunks — test if caveat is present
    review_chunks = [c for c in chunks if c.tag.default_behavior == "review"]
    remaining = max_count - len(pairs)
    for chunk in review_chunks[:remaining]:
        # Generate proper question with LLM
        caveat_q = ""
        if llm_call:
            try:
                prompt = (
                    f"This chunk in a {domain_type or 'general'} knowledge base is flagged "
                    f"as '{chunk.tag.value}' and will be served with a warning.\n\n"
                    f"CONTENT: {chunk.content[:400]}\n\n"
                    f"Write ONE natural question a customer would ask that this chunk answers.\n\n"
                    f"Return ONLY the question."
                )
                caveat_q = llm_call(prompt)
                if caveat_q:
                    caveat_q = caveat_q.strip().split("\n")[0].strip()
            except Exception:
                pass
        if not caveat_q:
            caveat_q = f"What can you tell me about {chunk.heading}?"

        pairs.append(QAPair(
            pair_id=f"qa_gov_{len(pairs) + 1}",
            question=caveat_q,
            expected_answer=_generate_expected_answer(caveat_q, chunk.content, llm_call),
            expected_behavior="caveat",
            source_chunk_id=chunk.chunk_id,
            source_chunk_heading=chunk.heading,
            source="governance",
            reasoning=f"Caveat probe: tagged as {chunk.tag.value}, served with warning",
            confidence="medium",
            must_review=True,
        ))

    return pairs


def _generate_clarity_probes(chunks: list[Chunk], a30_findings: list,
                             llm_call: Optional[Callable],
                             domain_type: str, max_count: int) -> list[QAPair]:
    """Source 3: test chunks where A30 fixed clarity issues."""
    pairs = []

    # Group fixes by chunk
    fixes_by_chunk = {}
    for f in (a30_findings or []):
        if f.fixed:
            fixes_by_chunk.setdefault(f.chunk_id, []).append(f)

    chunk_map = {c.chunk_id: c for c in chunks}

    for chunk_id, fixes in list(fixes_by_chunk.items())[:max_count]:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        fix_descriptions = []
        for f in fixes[:3]:
            if f.original_term and f.fix_detail:
                fix_descriptions.append(f'"{f.original_term}" -> {f.fix_detail}')
            elif f.original_term and f.proposed_fix:
                fix_descriptions.append(f'"{f.original_term}" -> "{f.proposed_fix}"')
        fixes_text = ", ".join(fix_descriptions) if fix_descriptions else "clarity improvements"

        question = ""
        if llm_call and fix_descriptions:
            try:
                prompt = (
                    f"This chunk had clarity issues that were fixed:\n"
                    f"FIXES: {fixes_text}\n\n"
                    f"CURRENT CONTENT (after fixes): {chunk.content[:400]}\n\n"
                    f"Write ONE question where the answer quality depends on these "
                    f"clarity fixes. If the fixes weren't applied, the answer would "
                    f"be ambiguous.\n\n"
                    f"Return ONLY the question."
                )
                question = llm_call(prompt)
                if question:
                    question = question.strip().split("\n")[0].strip()
            except Exception:
                pass

        if not question:
            question = f"Can you explain {chunk.heading}?"

        pairs.append(QAPair(
            pair_id=f"qa_clarity_{len(pairs) + 1}",
            question=question,
            expected_answer=_generate_expected_answer(question, chunk.content, llm_call),
            expected_behavior="answer",
            source_chunk_id=chunk.chunk_id,
            source_chunk_heading=chunk.heading,
            source="clarity",
            reasoning=f"Clarity probe: tests if fixes applied ({fixes_text[:80]})",
            confidence="medium",
        ))

    return pairs


def _generate_consistency_probes(chunks: list[Chunk], a32_findings: list,
                                 llm_call: Optional[Callable],
                                 domain_type: str, max_count: int) -> list[QAPair]:
    """Source 4: test contradictions.

    Question is natural (doesn't mention the conflict).
    Answer depends on user's A32 decision:
      - User picked winner → expected answer = winner's content
      - User kept both → expected answer = both options with caveat
    """
    pairs = []
    chunk_map = {c.chunk_id: c for c in chunks}
    seen_topics = set()  # dedup by conflict rationale topic

    for f in (a32_findings or [])[:max_count]:
        chunk_a = chunk_map.get(f.chunk_a_id)
        chunk_b = chunk_map.get(f.chunk_b_id)
        if not chunk_a or not chunk_b:
            continue

        # Dedup by conflict type + topic
        topic_key = f"{f.conflict_type}:{f.rationale[:40]}".lower()
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)

        # Determine winner and expected answer based on user decision
        if f.user_decision == "select_a":
            winner = chunk_a
            expected = _generate_expected_answer(
                "", winner.content, llm_call) if llm_call else _answer_from_content(winner.content)
            behavior = "answer"
            reasoning = (f"Consistency probe: {f.conflict_type} — "
                        f"user selected {chunk_a.heading} as correct")
        elif f.user_decision == "select_b":
            winner = chunk_b
            expected = _generate_expected_answer(
                "", winner.content, llm_call) if llm_call else _answer_from_content(winner.content)
            behavior = "answer"
            reasoning = (f"Consistency probe: {f.conflict_type} — "
                        f"user selected {chunk_b.heading} as correct")
        else:
            # keep_both or pending — show both options with caveat
            winner = chunk_a
            side_a = _answer_from_content(chunk_a.content)[:120]
            side_b = _answer_from_content(chunk_b.content)[:120]
            expected = (f"There are two perspectives: "
                       f"({chunk_a.heading}): {side_a}. "
                       f"({chunk_b.heading}): {side_b}.")
            behavior = "caveat"
            reasoning = (f"Consistency probe: {f.conflict_type} — "
                        f"both perspectives should be shown")

        # Generate natural question (doesn't mention the conflict)
        question = ""
        if llm_call:
            try:
                prompt = (
                    f"A customer is looking for information about this topic "
                    f"in a {domain_type or 'general'} knowledge base:\n\n"
                    f"TOPIC: {f.rationale}\n"
                    f"CONTENT: {_answer_from_content(winner.content)[:200]}\n\n"
                    f"Write ONE natural question a customer would ask about this topic. "
                    f"Do NOT mention any conflict or contradiction. "
                    f"Just ask a normal question.\n\n"
                    f"Return ONLY the question."
                )
                question = llm_call(prompt)
                if question:
                    question = question.strip().split("\n")[0].strip()
            except Exception:
                pass

        if not question:
            question = f"Can you tell me about {winner.heading}?"

        # Generate proper expected answer for resolved conflicts
        if behavior == "answer" and llm_call:
            expected = _generate_expected_answer(question, winner.content, llm_call)

        pairs.append(QAPair(
            pair_id=f"qa_consist_{len(pairs) + 1}",
            question=question,
            expected_answer=expected,
            expected_behavior=behavior,
            source_chunk_id=winner.chunk_id,
            source_chunk_heading=winner.heading,
            source="consistency",
            reasoning=reasoning,
            confidence="medium",
            must_review=True,
        ))

    return pairs


# =====================================================================
# Main generator
# =====================================================================

class QAGenerator:
    """A41 — Generate Q&A pairs that test Phase 1-3 outcomes."""

    def __init__(self, config: Optional[A41Config] = None):
        self.config = config or A41Config()

    def run(self, chunks: list[Chunk],
            a30_findings: Optional[list] = None,
            a31_findings: Optional[list] = None,
            a32_findings: Optional[list] = None,
            domain_context: Optional[DomainContext] = None) -> ModuleOutput:
        """Generate Q&A pairs from Phase 3 results."""
        t0 = time.perf_counter()
        domain_type = domain_context.domain_type if domain_context else ""
        llm = self.config.llm_call

        # Calculate recommended counts
        recommended = _recommend_counts(chunks, a30_findings or [],
                                        a31_findings or [], a32_findings or [])

        # Use user's total or recommended
        total = self.config.total_questions if self.config.total_questions > 0 else recommended["total"]

        # Fixed sources: governance, clarity, consistency — always full count
        # Variable source: topic — gets whatever remains
        budget = {"topic": 0, "governance": 0, "clarity": 0, "consistency": 0}

        if self.config.include_governance:
            budget["governance"] = recommended["governance"]
        if self.config.include_clarity:
            budget["clarity"] = recommended["clarity"]
        if self.config.include_consistency:
            budget["consistency"] = recommended["consistency"]

        fixed_total = budget["governance"] + budget["clarity"] + budget["consistency"]

        if self.config.include_topic:
            budget["topic"] = max(0, total - fixed_total)

        # Warn if total is less than fixed sources
        if total < fixed_total:
            # Can't fit all fixed sources — topic gets 0, fixed stays
            pass

        # Generate
        all_pairs = []

        if budget["topic"] > 0:
            all_pairs.extend(_generate_topic_questions(
                chunks, llm, domain_type, budget["topic"]))

        if budget["governance"] > 0:
            all_pairs.extend(_generate_governance_probes(
                chunks, a31_findings or [], llm, domain_type, budget["governance"]))

        if budget["clarity"] > 0:
            all_pairs.extend(_generate_clarity_probes(
                chunks, a30_findings or [], llm, domain_type, budget["clarity"]))

        if budget["consistency"] > 0:
            all_pairs.extend(_generate_consistency_probes(
                chunks, a32_findings or [], llm, domain_type, budget["consistency"]))

        # Merge user-provided Q&A pairs
        if self.config.user_qa_pairs:
            for i, uqa in enumerate(self.config.user_qa_pairs):
                if not isinstance(uqa, dict) or "question" not in uqa:
                    continue
                all_pairs.append(QAPair(
                    pair_id=f"qa_user_{i + 1}",
                    question=uqa["question"],
                    expected_answer=uqa.get("expected_answer", ""),
                    expected_behavior=uqa.get("expected_behavior", "answer"),
                    source="user",
                    reasoning="User-provided test question",
                    confidence="high",
                    must_review=False,
                ))

        # Dedup by question
        seen = set()
        unique = []
        for p in all_pairs:
            key = p.question.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(p)
        all_pairs = unique

        # Coverage stats
        from collections import Counter
        covered_ids = {p.source_chunk_id for p in all_pairs if p.source_chunk_id}
        by_source = dict(Counter(p.source for p in all_pairs))
        by_behavior = dict(Counter(p.expected_behavior for p in all_pairs))

        qa_set = QASet(
            pairs=all_pairs,
            total_chunks=len(chunks),
            chunks_covered=len(covered_ids),
            by_source=by_source,
            by_behavior=by_behavior,
            recommended_total=recommended["total"],
        )

        return ModuleOutput(
            module_id="A41",
            module_name="Q&A Generator",
            detected=len(all_pairs),
            resolved=sum(1 for p in all_pairs if not p.must_review),
            remaining=sum(1 for p in all_pairs if p.must_review),
            words_in=sum(c.words for c in chunks),
            words_out=sum(c.words for c in chunks),
            elapsed_seconds=time.perf_counter() - t0,
            data={"qa_set": qa_set, "recommended": recommended},
        )
