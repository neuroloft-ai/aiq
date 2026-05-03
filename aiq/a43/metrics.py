"""A43 — Metrics

Computes quality scores from A42 retrieval test results. All metrics are
computed for both "before" (raw chunks) and "after" (pipeline chunks),
showing the delta = value delivered by AIQ.

What it computes:
    1. Recall — correct answers / total (threshold-based, before vs after)
    2. Accuracy — mean similarity score across all questions
    3. Safety — answer_safe tests passing (remediated content served correctly)
    4. Hallucination — wrong answers for caveat questions (should abstain)
    5. Risk score — failure rate relative to user-defined tolerance
    6. By source — breakdown per question type (topic/governance/clarity/consistency)
    7. Per-module — detected / auto-resolved / user-resolved / pending per module

How it works:
    - Verdict-to-score conversion: correct=1.0, partial=0.5, incorrect=0.0
    - Recall threshold: score >= threshold counts as "correct" (default: 0.7)
    - Risk: (failure_rate / tolerance) * 50, capped at 100. Score 50 = at tolerance.
    - Per-module issues collected from session_modules dict (A12-A32 outputs)
    - recalculate_risk() allows re-computing risk with different tolerance

Config:
    threshold: similarity score threshold for "correct" verdict (default: 0.7)
    risk_tolerance: acceptable failure rate (default: 0.01 = 1%)

Config exposed to AIQConfig:
    eval_threshold      -> A43Config.threshold       (default: 0.7)
    eval_risk_tolerance -> A43Config.risk_tolerance   (default: 0.01)

Auto-detected (no user input needed):
    All metrics computed from A42 PairResult data
    Per-module issues collected from module outputs

Input:  list[PairResult] from A42, optional session_modules dict, optional chunks
Output: ModuleOutput with .data["metrics_result"] = MetricsResult

LLM required: No. Pure computation — no LLM, no embeddings, no external calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from aiq.core.types import ModuleOutput


# =====================================================================
# Types
# =====================================================================

@dataclass
class RecallMetrics:
    """Recall = correct / total."""
    total: int = 0
    correct: int = 0
    rate: float = 0.0


@dataclass
class AccuracyMetrics:
    """Accuracy = mean similarity score."""
    total: int = 0
    mean_score: float = 0.0
    scores: list = field(default_factory=list)


@dataclass
class SafetyMetrics:
    """Safety = answer_safe tests passing."""
    total: int = 0
    passed: int = 0


@dataclass
class HallucinationMetrics:
    """Hallucination = wrong answers when should abstain."""
    total: int = 0
    hallucinated: int = 0


@dataclass
class RiskScore:
    """Risk relative to tolerance."""
    failure_rate: float = 0.0
    tolerance: float = 0.01
    score: float = 0.0
    total: int = 0
    failures: int = 0


@dataclass
class SourceBreakdown:
    """Per-source recall and accuracy."""
    total: int = 0
    test1_correct: int = 0
    test2_correct: int = 0
    test1_accuracy: float = 0.0
    test2_accuracy: float = 0.0


@dataclass
class ModuleIssues:
    """Per-module issue tracking."""
    module_id: str
    module_name: str
    detected: int = 0
    auto_resolved: int = 0
    user_resolved: int = 0
    pending: int = 0


@dataclass
class MetricsResult:
    """Full metrics output."""
    # Recall (before vs after)
    recall_before: RecallMetrics = field(default_factory=RecallMetrics)
    recall_after: RecallMetrics = field(default_factory=RecallMetrics)

    # Accuracy (before vs after)
    accuracy_before: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    accuracy_after: AccuracyMetrics = field(default_factory=AccuracyMetrics)

    # Safety (before vs after)
    safety_before: SafetyMetrics = field(default_factory=SafetyMetrics)
    safety_after: SafetyMetrics = field(default_factory=SafetyMetrics)

    # Hallucination (before vs after)
    halluc_before: HallucinationMetrics = field(default_factory=HallucinationMetrics)
    halluc_after: HallucinationMetrics = field(default_factory=HallucinationMetrics)

    # Risk (before vs after)
    risk_before: RiskScore = field(default_factory=RiskScore)
    risk_after: RiskScore = field(default_factory=RiskScore)

    # By source
    by_source: dict = field(default_factory=dict)  # source -> SourceBreakdown

    # Per-module issues
    module_issues: list = field(default_factory=list)

    # Threshold used
    threshold: float = 0.7


# =====================================================================
# Config
# =====================================================================

@dataclass
class A43Config:
    """Configuration for metrics."""
    threshold: float = 0.7        # score >= threshold = correct
    risk_tolerance: float = 0.01  # 1% default


# =====================================================================
# Score conversion
# =====================================================================

def _verdict_to_score(verdict: str) -> float:
    """Convert LLM verdict to similarity score.

    For now uses simple mapping. Later will be replaced with
    fact-based scoring (matched_facts / total_facts).
    """
    if verdict == "correct":
        return 1.0
    if verdict == "partial":
        return 0.5
    return 0.0


# =====================================================================
# Computation
# =====================================================================

def _compute_recall(pair_results: list, pool: str, threshold: float) -> RecallMetrics:
    """Recall = questions with score >= threshold / total."""
    total = len(pair_results)
    if total == 0:
        return RecallMetrics()

    correct = 0
    for r in pair_results:
        judge = r.test1_raw if pool == "before" else r.test2_pipeline
        if judge:
            score = _verdict_to_score(judge.verdict)
            if score >= threshold:
                correct += 1

    return RecallMetrics(
        total=total,
        correct=correct,
        rate=correct / total,
    )


def _compute_accuracy(pair_results: list, pool: str) -> AccuracyMetrics:
    """Accuracy = mean similarity score across all questions."""
    total = len(pair_results)
    if total == 0:
        return AccuracyMetrics()

    scores = []
    for r in pair_results:
        judge = r.test1_raw if pool == "before" else r.test2_pipeline
        if judge:
            scores.append(_verdict_to_score(judge.verdict))
        else:
            scores.append(0.0)

    return AccuracyMetrics(
        total=total,
        mean_score=sum(scores) / total,
        scores=scores,
    )


def _compute_safety(pair_results: list, pool: str, threshold: float) -> SafetyMetrics:
    """Safety = answer_safe questions passing."""
    safe_pairs = [r for r in pair_results if r.expected_behavior == "answer_safe"]
    total = len(safe_pairs)
    if total == 0:
        return SafetyMetrics()

    passed = 0
    for r in safe_pairs:
        judge = r.test1_raw if pool == "before" else r.test2_pipeline
        if judge and _verdict_to_score(judge.verdict) >= threshold:
            passed += 1

    return SafetyMetrics(total=total, passed=passed)


def _compute_hallucination(pair_results: list, pool: str) -> HallucinationMetrics:
    """Hallucination = wrong answers for caveat/no_answer questions.

    When system should say 'I don't know' or show caveat but gives wrong answer.
    """
    caveat_pairs = [r for r in pair_results if r.expected_behavior == "caveat"]
    total = len(caveat_pairs)
    if total == 0:
        return HallucinationMetrics()

    hallucinated = 0
    for r in caveat_pairs:
        judge = r.test1_raw if pool == "before" else r.test2_pipeline
        if judge and _verdict_to_score(judge.verdict) == 0.0:
            hallucinated += 1

    return HallucinationMetrics(total=total, hallucinated=hallucinated)


def _compute_risk(pair_results: list, pool: str, threshold: float,
                  tolerance: float) -> RiskScore:
    """Risk = failure rate relative to tolerance."""
    total = len(pair_results)
    if total == 0:
        return RiskScore(tolerance=tolerance)

    failures = 0
    for r in pair_results:
        judge = r.test1_raw if pool == "before" else r.test2_pipeline
        if not judge or _verdict_to_score(judge.verdict) < threshold:
            failures += 1

    failure_rate = failures / total

    if tolerance <= 0:
        score = 0.0 if failures == 0 else 100.0
    else:
        score = min(100.0, (failure_rate / tolerance) * 50.0)

    return RiskScore(
        failure_rate=failure_rate,
        tolerance=tolerance,
        score=score,
        total=total,
        failures=failures,
    )


def _compute_by_source(pair_results: list, threshold: float) -> dict:
    """Breakdown per source type."""
    sources = {}
    for r in pair_results:
        src = getattr(r, 'source', 'unknown')
        if src not in sources:
            sources[src] = {"pairs": []}
        sources[src]["pairs"].append(r)

    result = {}
    for src, data in sources.items():
        pairs = data["pairs"]
        total = len(pairs)

        t1_scores = []
        t2_scores = []
        for r in pairs:
            s1 = _verdict_to_score(r.test1_raw.verdict) if r.test1_raw else 0.0
            s2 = _verdict_to_score(r.test2_pipeline.verdict) if r.test2_pipeline else 0.0
            t1_scores.append(s1)
            t2_scores.append(s2)

        result[src] = SourceBreakdown(
            total=total,
            test1_correct=sum(1 for s in t1_scores if s >= threshold),
            test2_correct=sum(1 for s in t2_scores if s >= threshold),
            test1_accuracy=sum(t1_scores) / total if total else 0,
            test2_accuracy=sum(t2_scores) / total if total else 0,
        )

    return result


def _collect_module_issues(session_modules: dict, chunks: list = None) -> list[ModuleIssues]:
    """Collect per-module issue counts with auto/user/pending breakdown."""
    issues = []

    # A12 Normalize
    a12 = session_modules.get("a12_output")
    if a12:
        gaps = a12.remaining
        issues.append(ModuleIssues(
            module_id="A12", module_name="Normalize",
            detected=a12.detected, auto_resolved=a12.resolved,
            user_resolved=0, pending=gaps,
        ))

    # A13 Structure
    a13 = session_modules.get("a13_output")
    if a13:
        issues.append(ModuleIssues(
            module_id="A13", module_name="Structure",
            detected=a13.detected, auto_resolved=a13.resolved,
            user_resolved=0, pending=a13.remaining,
        ))

    # A30 Clarity
    a30 = session_modules.get("a30_output")
    if a30:
        auto = sum(1 for f in a30.findings if f.fixed and 'acronym' in f.issue_type)
        a30_confirmed = session_modules.get("_a30_confirmed", False)
        unfixed = [f for f in a30.findings if not f.fixed]
        unfixed_with_fix = sum(1 for f in unfixed if f.proposed_fix)
        unfixed_no_fix = sum(1 for f in unfixed if not f.proposed_fix)

        if a30_confirmed:
            # User confirmed — accepted suggestions become user_resolved, skipped become caveats (also resolved)
            user = sum(1 for f in a30.findings if f.fixed and 'acronym' not in f.issue_type) + len(unfixed)
            pending = 0
        else:
            user = sum(1 for f in a30.findings if f.fixed and 'acronym' not in f.issue_type)
            pending = len(unfixed)

        issues.append(ModuleIssues(
            module_id="A30", module_name="Semantic Clarity",
            detected=a30.detected, auto_resolved=auto,
            user_resolved=user, pending=pending,
        ))

    # A31 Governance
    a31 = session_modules.get("a31_output")
    if a31:
        findings = a31.findings if hasattr(a31, 'findings') else []
        detected = len(findings)

        # Group findings by chunk to match with chunk-level actions
        chunk_ids_remediated = set()
        chunk_ids_review = set()
        if chunks:
            for c in chunks:
                if "remediated" in (c.tag_module or ""):
                    chunk_ids_remediated.add(c.chunk_id)
                elif c.tag.default_behavior == "review":
                    chunk_ids_review.add(c.chunk_id)

        # Count findings per resolution type
        auto_resolved = sum(1 for f in findings if f.chunk_id in chunk_ids_remediated)
        review_findings = sum(1 for f in findings if f.chunk_id in chunk_ids_review)

        a31_confirmed = session_modules.get("_a31_confirmed", False)
        if a31_confirmed:
            user_resolved = review_findings
            pending = 0
        else:
            user_resolved = 0
            pending = review_findings

        # Remaining findings not in remediated or review chunks
        other = detected - auto_resolved - review_findings
        auto_resolved += max(0, other)  # tagged but not remediated/review = auto-handled

        issues.append(ModuleIssues(
            module_id="A31", module_name="Content Governance",
            detected=detected, auto_resolved=auto_resolved,
            user_resolved=user_resolved, pending=pending,
        ))

    # A32 Consistency
    a32 = session_modules.get("a32_output")
    if a32:
        a32_confirmed = session_modules.get("_a32_confirmed", False)
        decided = sum(1 for f in a32.findings if f.user_decision != "pending")
        still_pending = sum(1 for f in a32.findings if f.user_decision == "pending")

        if a32_confirmed:
            # User confirmed — all become user_resolved (even pending ones become caveats)
            user_resolved = len(a32.findings)
            pending = 0
        else:
            user_resolved = decided
            pending = still_pending

        issues.append(ModuleIssues(
            module_id="A32", module_name="Consistency",
            detected=a32.detected, auto_resolved=0,
            user_resolved=user_resolved, pending=pending,
        ))

    return issues


# =====================================================================
# Main calculator
# =====================================================================

class MetricsCalculator:
    """A43 — Compute metrics from A42 results."""

    def __init__(self, config: Optional[A43Config] = None):
        self.config = config or A43Config()

    def run(self, pair_results: list,
            session_modules: Optional[dict] = None,
            chunks: Optional[list] = None) -> ModuleOutput:
        """Compute all metrics."""
        threshold = self.config.threshold
        tolerance = self.config.risk_tolerance

        result = MetricsResult(
            recall_before=_compute_recall(pair_results, "before", threshold),
            recall_after=_compute_recall(pair_results, "after", threshold),
            accuracy_before=_compute_accuracy(pair_results, "before"),
            accuracy_after=_compute_accuracy(pair_results, "after"),
            safety_before=_compute_safety(pair_results, "before", threshold),
            safety_after=_compute_safety(pair_results, "after", threshold),
            halluc_before=_compute_hallucination(pair_results, "before"),
            halluc_after=_compute_hallucination(pair_results, "after"),
            risk_before=_compute_risk(pair_results, "before", threshold, tolerance),
            risk_after=_compute_risk(pair_results, "after", threshold, tolerance),
            by_source=_compute_by_source(pair_results, threshold),
            module_issues=_collect_module_issues(session_modules or {}, chunks),
            threshold=threshold,
        )

        return ModuleOutput(
            module_id="A43",
            module_name="Metrics",
            detected=len(pair_results),
            resolved=0, remaining=0,
            words_in=0, words_out=0,
            data={"metrics_result": result},
        )

    def recalculate_risk(self, pair_results: list, new_tolerance: float) -> tuple[RiskScore, RiskScore]:
        """Recalculate risk with new tolerance."""
        before = _compute_risk(pair_results, "before", self.config.threshold, new_tolerance)
        after = _compute_risk(pair_results, "after", self.config.threshold, new_tolerance)
        return before, after
