"""A42 — Retrieval Test

Runs each Q&A pair against two chunk pools (before/after pipeline) and
judges whether the retrieved content correctly answers the question.

What it does:
    For each Q&A pair from A41:
    1. Test 1: search A10 raw chunks (before pipeline) for the answer
    2. Test 2: search Phase 3 pipeline chunks (after processing) for the answer
    3. Judge each retrieval: correct / partial / incorrect

    Comparison is CONTENT-based (not chunk ID matching). A better embedding
    model won't inflate scores — the answer text must actually be present.

How it works:
    Retrieval methods:
    - BM25: term frequency scoring (no external dependencies)
    - Cosine: embedding-based similarity (requires embed_fn)

    Judging methods:
    - cosine: word overlap or embedding similarity against expected answer
    - llm: LLM judges if retrieved content answers the question
    - cosine_then_llm: cosine first, LLM for borderline "partial" cases

    Pipeline chunks are filtered to "servable" only — blocked chunks excluded.
    For "block" expected_behavior: any retrieval is marked incorrect (content
    should not be findable).

Config:
    retrieval_method: "cosine" | "bm25" (default: "cosine")
    judge_method: "llm" | "cosine" | "cosine_then_llm" (default: "llm")
    top_k: number of chunks to retrieve per query (default: 3)
    cosine_correct: similarity threshold for "correct" verdict (default: 0.80)
    cosine_partial: similarity threshold for "partial" verdict (default: 0.60)
    embed_fn: callable(list[str]) -> list[list[float]] for cosine retrieval
    llm_call: callable(prompt: str) -> str for LLM judging

Config exposed to AIQConfig:
    eval_retrieval_method -> A42Config.retrieval_method (default: "cosine")
    eval_judge_method     -> A42Config.judge_method     (default: "llm")
    eval_top_k            -> A42Config.top_k            (default: 3)
    (embed_fn and llm_call wired from AIQConfig)

Auto-detected (no user input needed):
    Servable chunk filtering — based on chunk tags (blocked chunks excluded)
    Verdict — based on content comparison against expected answer

    Future: support tag_behavior overrides for servable filtering.

Input:  list[QAPair], list[Chunk] (raw), list[Chunk] (pipeline)
Output: ModuleOutput with .data["test_result"] = TestResult containing list[PairResult]

LLM required: No (BM25 + word-overlap cosine works without LLM/embeddings).
    LLM enhances: embedding-based retrieval, semantic judging.
"""
from __future__ import annotations

import re
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import Chunk, ModuleOutput


# =====================================================================
# Types
# =====================================================================

@dataclass
class JudgeResult:
    """Judgment for one retrieval test."""
    verdict: str            # "correct" | "incorrect" | "partial"
    reasoning: str          # why this verdict
    retrieved_content: str  # what was actually retrieved (for display)
    retrieved_heading: str  # heading of retrieved chunk
    score: float = 0.0     # similarity score
    method: str = ""       # "cosine" | "llm" | "cosine_then_llm"


@dataclass
class PairResult:
    """Full result for one Q&A pair — test 1 (raw) and test 2 (pipeline)."""
    pair_id: str
    question: str
    expected_answer: str
    expected_behavior: str
    reasoning: str           # from A41 — what we're testing
    source: str              # topic | governance | clarity | consistency

    test1_raw: Optional[JudgeResult] = None
    test2_pipeline: Optional[JudgeResult] = None


@dataclass
class TestResult:
    """Full result set from A42."""
    pair_results: list = field(default_factory=list)
    total_pairs: int = 0
    test1_correct: int = 0
    test2_correct: int = 0
    elapsed_seconds: float = 0.0


# =====================================================================
# Config
# =====================================================================

@dataclass
class A42Config:
    """Configuration for retrieval testing."""
    retrieval_method: str = "cosine"     # "cosine" | "bm25"
    judge_method: str = "llm"           # "llm" | "cosine" | "cosine_then_llm"
    top_k: int = 3
    # Cosine thresholds
    cosine_correct: float = 0.80
    cosine_partial: float = 0.60
    # Functions
    embed_fn: Optional[Callable] = None   # (list[str]) -> list[list[float]]
    llm_call: Optional[Callable] = None


# =====================================================================
# Retrieval — BM25
# =====================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _bm25_score(query_tokens, doc_tokens, avg_dl, k1=1.5, b=0.75):
    from collections import Counter
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        f = tf.get(qt, 0)
        if f == 0:
            continue
        score += (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


def _retrieve_bm25(question: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []
    docs = [(c, _tokenize(c.content)) for c in chunks]
    avg_dl = sum(len(d) for _, d in docs) / max(len(docs), 1)
    scored = [(c, _bm25_score(query_tokens, dt, avg_dl)) for c, dt in docs]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# =====================================================================
# Retrieval — Cosine
# =====================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _retrieve_cosine(question: str, chunks: list[Chunk],
                     chunk_embeddings: dict, query_emb: list[float],
                     top_k: int) -> list[tuple[Chunk, float]]:
    scored = []
    for c in chunks:
        emb = chunk_embeddings.get(c.chunk_id)
        if emb is None:
            continue
        scored.append((c, _cosine_sim(query_emb, emb)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# =====================================================================
# Judging
# =====================================================================

def _judge_cosine(expected: str, retrieved: str,
                  config: A42Config,
                  expected_emb=None, retrieved_emb=None) -> JudgeResult:
    """Judge by cosine similarity between expected and retrieved."""
    if expected_emb and retrieved_emb:
        sim = _cosine_sim(expected_emb, retrieved_emb)
    else:
        # Word overlap fallback
        exp_words = set(_tokenize(expected))
        ret_words = set(_tokenize(retrieved))
        if not exp_words:
            sim = 1.0
        else:
            sim = len(exp_words & ret_words) / len(exp_words)

    if sim >= config.cosine_correct:
        verdict = "correct"
    elif sim >= config.cosine_partial:
        verdict = "partial"
    else:
        verdict = "incorrect"

    return JudgeResult(
        verdict=verdict,
        reasoning=f"Cosine similarity: {sim:.2f}",
        retrieved_content=retrieved[:500],
        retrieved_heading="",
        score=sim,
        method="cosine",
    )


def _judge_llm(question: str, expected: str, retrieved: str,
               behavior: str, a41_reasoning: str,
               llm_call: Callable) -> JudgeResult:
    """LLM judges if retrieved content correctly answers the question."""
    prompt = f"""You are judging a RAG retrieval system's response.

QUESTION: {question}

EXPECTED ANSWER: {expected[:800]}

RETRIEVED CONTENT: {retrieved[:1200]}

TEST CONTEXT: {a41_reasoning[:200]}

JUDGE the retrieved content against the expected answer.

VERDICT RULES:
- "correct": the retrieved content contains the KEY facts needed to answer the question.
  It does NOT need to be word-for-word identical. If 70%+ of the important information is present, mark as correct.
  Minor missing details (e.g., one fee amount, one sub-step) are OK for correct.
- "partial": the retrieved content is on the RIGHT TOPIC but is missing MAJOR parts of the answer (less than 50% of key facts present).
- "incorrect": the retrieved content is about a DIFFERENT TOPIC entirely, or contains problematic content (PII, internal notes).

IMPORTANT:
- Focus on whether the retrieved content could help a customer get the right answer, not whether it matches exactly.
- If the retrieved content covers the main point of the expected answer, that is "correct" even if some minor details differ.
- Only use "partial" when significant information is genuinely missing, not for minor differences.

SPECIAL CHECKS based on test context:
- Governance/PII test: if personal names, personal emails, or personal phone numbers appear → "incorrect" (PII leaked)
- Figure test: if figure description is present and matches → "correct"
- Clarity test: check if the fix was applied (pronouns resolved, acronyms expanded)
- Consistency test: check if the answer matches the expected version

Return as JSON:
{{"verdict": "correct|partial|incorrect", "reasoning": "one sentence explanation"}}

Return ONLY the JSON."""

    try:
        result = llm_call(prompt)
        if result:
            import json
            text = result.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            return JudgeResult(
                verdict=parsed.get("verdict", "incorrect"),
                reasoning=parsed.get("reasoning", ""),
                retrieved_content=retrieved[:500],
                retrieved_heading="",
                score=0.0,
                method="llm",
            )
    except Exception:
        pass

    # Fallback to word overlap
    return _judge_cosine(expected, retrieved, A42Config())


def _judge(question: str, expected: str, retrieved_chunks: list[tuple[Chunk, float]],
           behavior: str, a41_reasoning: str,
           config: A42Config,
           answer_emb=None, chunk_embeddings=None) -> JudgeResult:
    """Judge retrieval results for one question."""
    if not retrieved_chunks:
        return JudgeResult(
            verdict="incorrect",
            reasoning="No chunks retrieved",
            retrieved_content="",
            retrieved_heading="",
        )

    top1_chunk, top1_score = retrieved_chunks[0]
    retrieved_content = top1_chunk.content
    retrieved_heading = top1_chunk.heading

    # For "block" behavior: any retrieval is a failure
    if behavior == "block":
        return JudgeResult(
            verdict="incorrect",
            reasoning="Content should be blocked but was retrievable",
            retrieved_content=retrieved_content[:500],
            retrieved_heading=retrieved_heading,
            score=top1_score,
        )

    # Judge content match
    if config.judge_method == "llm" and config.llm_call:
        result = _judge_llm(
            question, expected, retrieved_content,
            behavior, a41_reasoning, config.llm_call)
    elif config.judge_method == "cosine_then_llm":
        # Cosine first, LLM for unclear
        cosine_result = _judge_cosine(
            expected, retrieved_content, config,
            answer_emb, chunk_embeddings.get(top1_chunk.chunk_id) if chunk_embeddings else None)
        if cosine_result.verdict == "partial" and config.llm_call:
            result = _judge_llm(
                question, expected, retrieved_content,
                behavior, a41_reasoning, config.llm_call)
        else:
            result = cosine_result
    else:
        result = _judge_cosine(
            expected, retrieved_content, config,
            answer_emb, chunk_embeddings.get(top1_chunk.chunk_id) if chunk_embeddings else None)

    result.retrieved_heading = retrieved_heading
    result.score = top1_score
    return result


# =====================================================================
# Embedding helpers
# =====================================================================

def _embed_batch(texts: list[str], embed_fn: Callable) -> list[list[float]]:
    batch_size = 50
    all_embs = []
    for i in range(0, len(texts), batch_size):
        all_embs.extend(embed_fn(texts[i:i + batch_size]))
    return all_embs


# =====================================================================
# Main tester
# =====================================================================

class RetrievalTester:
    """A42 — Run retrieval tests on raw and pipeline chunks."""

    def __init__(self, config: Optional[A42Config] = None):
        self.config = config or A42Config()

    def run(self, qa_pairs: list,
            raw_chunks: list[Chunk],
            pipeline_chunks: list[Chunk]) -> ModuleOutput:
        """Run both tests for all Q&A pairs.

        Args:
            qa_pairs: from A41 QASet.pairs
            raw_chunks: from A10 (before pipeline)
            pipeline_chunks: from A14 (after Phase 3)
        """
        t0 = time.perf_counter()

        # Only search servable pipeline chunks (skip any remaining blocked)
        servable = [c for c in pipeline_chunks if c.tag.default_behavior != "block"]

        # Build embeddings if cosine
        raw_embs = {}
        pipe_embs = {}
        query_embs = {}
        answer_embs = {}

        if self.config.retrieval_method == "cosine" and self.config.embed_fn:
            raw_embs, pipe_embs, query_embs, answer_embs = self._build_embeddings(
                qa_pairs, raw_chunks, servable)

        pair_results = []
        test1_correct = 0
        test2_correct = 0

        for pair in qa_pairs:
            q_emb = query_embs.get(pair.pair_id)
            a_emb = answer_embs.get(pair.pair_id)

            # Test 1: search raw chunks
            raw_results = self._retrieve(pair.question, raw_chunks, raw_embs, q_emb)
            test1 = _judge(
                pair.question, pair.expected_answer, raw_results,
                pair.expected_behavior, pair.reasoning,
                self.config, a_emb, raw_embs,
            )

            # Test 2: search pipeline chunks (servable only)
            pipe_results = self._retrieve(pair.question, servable, pipe_embs, q_emb)
            test2 = _judge(
                pair.question, pair.expected_answer, pipe_results,
                pair.expected_behavior, pair.reasoning,
                self.config, a_emb, pipe_embs,
            )

            if test1.verdict == "correct":
                test1_correct += 1
            if test2.verdict == "correct":
                test2_correct += 1

            pair_results.append(PairResult(
                pair_id=pair.pair_id,
                question=pair.question,
                expected_answer=pair.expected_answer,
                expected_behavior=pair.expected_behavior,
                reasoning=pair.reasoning,
                source=pair.source,
                test1_raw=test1,
                test2_pipeline=test2,
            ))

        test_result = TestResult(
            pair_results=pair_results,
            total_pairs=len(pair_results),
            test1_correct=test1_correct,
            test2_correct=test2_correct,
            elapsed_seconds=time.perf_counter() - t0,
        )

        return ModuleOutput(
            module_id="A42",
            module_name="Retrieval Test",
            detected=len(pair_results),
            resolved=test2_correct,
            remaining=len(pair_results) - test2_correct,
            words_in=0, words_out=0,
            elapsed_seconds=time.perf_counter() - t0,
            data={"test_result": test_result},
        )

    def _retrieve(self, question, chunks, embeddings, q_emb):
        if self.config.retrieval_method == "cosine" and q_emb and embeddings:
            return _retrieve_cosine(question, chunks, embeddings, q_emb, self.config.top_k)
        return _retrieve_bm25(question, chunks, self.config.top_k)

    def _build_embeddings(self, qa_pairs, raw_chunks, pipe_chunks):
        ef = self.config.embed_fn
        raw_texts = [c.content for c in raw_chunks]
        raw_ids = [c.chunk_id for c in raw_chunks]
        pipe_texts = [c.content for c in pipe_chunks]
        pipe_ids = [c.chunk_id for c in pipe_chunks]
        queries = [p.question for p in qa_pairs]
        q_ids = [p.pair_id for p in qa_pairs]
        answers = [p.expected_answer for p in qa_pairs]
        a_ids = [p.pair_id for p in qa_pairs]

        all_texts = raw_texts + pipe_texts + queries + answers
        all_embs = _embed_batch(all_texts, ef)

        idx = 0
        raw_e = dict(zip(raw_ids, all_embs[idx:idx+len(raw_ids)])); idx += len(raw_ids)
        pipe_e = dict(zip(pipe_ids, all_embs[idx:idx+len(pipe_ids)])); idx += len(pipe_ids)
        q_e = dict(zip(q_ids, all_embs[idx:idx+len(q_ids)])); idx += len(q_ids)
        a_e = dict(zip(a_ids, all_embs[idx:idx+len(a_ids)]))
        return raw_e, pipe_e, q_e, a_e
