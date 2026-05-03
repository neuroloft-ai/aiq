"""A32 — Consistency Checking

Finds contradictions between chunks using a three-layer architecture.
Contradictions are flagged for user decision — the system suggests a
winner but never auto-resolves.

What it detects:
    - Numeric contradictions: same unit + similar context, different values
      (e.g. "refunds take 5 days" vs "refunds take 10 days")
    - Authority contradictions: same responsibility, different actors
      (e.g. "billing handles refunds" vs "finance handles refunds")
    - Process contradictions: same task, different step sequences
    - Within-chunk authority conflicts (same chunk, two actors claim same role)

How it works:
    Layer A — Coverage: selects candidate pairs worth comparing
      - Adjacent chunks (within max_adjacency_distance)
      - Shared heading keywords (min_heading_overlap)
      - Shared domain anchor topics (from A11)
      Note: O(n^2) for pair generation. Works for single documents (~50 chunks).
      For large multi-document KBs, use embedding-based pair selection (future).

    Layer B — Detection: rule-based detectors on each pair
      - Numeric: extracts (number, unit, context) tuples, compares across pairs
      - Authority: extracts (actor, responsibility) from DomainContext actors
      - Process: compares numbered step sequences with similar headings

    Layer C — Resolution:
      - Date-aware winner suggestion using A22 metadata (page age, content dates)
      - Optional LLM judge: validates findings, removes false positives, suggests winners
      - User decision: select_a, select_b, keep_both, or bulk accept

Config:
    min_heading_overlap: shared heading words for pairing (default: 1)
    max_adjacency_distance: adjacent chunk pairing distance (default: 5)
    numeric_min_context_overlap: shared context words threshold (default: 0.3)
    use_llm_judge: enable LLM validation (default: False)
    llm_call: optional callable(prompt: str) -> str

Config exposed to AIQConfig:
    consistency_llm_judge  -> A32Config.use_llm_judge  (default: False)
    detection_confidence   -> filters findings before surfacing (future: numeric scoring)
    priority_authors       -> prefer authoritative authors in winner suggestion (future)
    scope_filters          -> suppress false positives across scopes (future)
    (llm_call wired from AIQConfig.llm_client)

Auto-detected (no user input needed):
    Candidate pairs — from adjacency, heading overlap, anchor topics
    Numeric facts — extracted from content via regex
    Authority claims — extracted using DomainContext actors
    Winner suggestion — from A22 page age and content dates
    Scope filtering — uses A11 scope qualifiers to suppress cross-scope false positives

    Each finding has confidence: "high" / "medium" / "low"
    Future: numeric confidence_score for threshold-based filtering.

Known limitations:
    - Numeric detection requires 2+ shared specific context words — may miss
      contradictions with slightly different phrasing
    - Pair selection is O(n^2) — breaks at scale (1000+ chunks)
    - Cross-scope filtering depends on scope qualifiers being in chunk content
    Future: embedding-based pair selection for scale + cross-document detection

Input:  list[Chunk], optional DomainContext
Output: ModuleOutput with .findings = list[ConsistencyFinding]

LLM required: No (all detectors are rule-based).
    LLM enhances: validates findings, removes false positives, suggests winners.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from aiq.core.types import Chunk, DomainContext, ModuleOutput


# =====================================================================
# Config
# =====================================================================

@dataclass
class A32Config:
    """Configuration for consistency checking."""
    # Candidate pair selection
    min_heading_overlap: int = 1         # min shared heading words for pairing
    max_adjacency_distance: int = 5      # adjacent chunks get paired
    # Numeric detection
    numeric_min_context_overlap: float = 0.3  # shared words between contexts
    # LLM judge (optional)
    use_llm_judge: bool = False
    llm_call: Optional[callable] = None


# =====================================================================
# Finding dataclass
# =====================================================================

@dataclass
class ConsistencyFinding:
    """One consistency issue between two chunks."""
    finding_id: str
    chunk_a_id: str
    chunk_b_id: str
    conflict_type: str      # numeric, authority, process, drift, superseded
    evidence_a: str
    evidence_b: str
    rationale: str
    suggested_winner: str   # "a", "b", or "" if no suggestion
    suggestion_reason: str  # "chunk_b is newer (March 2026 vs January 2022)"

    confidence: str = "high"   # "high" | "medium" | "low" — from LLM validation

    # User decision (set later)
    user_decision: str = "pending"  # pending, select_a, select_b, keep_both


# =====================================================================
# Layer A — Coverage engine
# =====================================================================

def _heading_keywords(heading: str) -> set:
    """Extract meaningful words from heading."""
    words = re.findall(r'\b[a-z]{3,}\b', heading.lower())
    stop = {"the", "and", "for", "with", "from", "updated", "new", "old", "our"}
    return {w for w in words if w not in stop}


def _build_candidate_pairs(chunks: list[Chunk], config: A32Config,
                           domain_context: Optional[DomainContext] = None) -> list[tuple[int, int]]:
    """Build list of chunk index pairs worth comparing.

    Pairs included if:
      - Adjacent (within max_adjacency_distance)
      - Share heading keywords
      - Share domain anchor topic
    """
    pairs = set()
    n = len(chunks)

    # Get domain anchors for topic overlap
    anchor_words = set()
    if domain_context:
        for anchor, variants in domain_context.domain_anchors.items():
            anchor_words.update(variants)

    for i in range(n):
        # Skip tagged chunks that shouldn't participate in consistency
        if chunks[i].tag.value in ("placeholder", "editorial"):
            continue

        keywords_i = _heading_keywords(chunks[i].heading)

        for j in range(i + 1, n):
            if chunks[j].tag.value in ("placeholder", "editorial"):
                continue

            # Adjacency
            if j - i <= config.max_adjacency_distance:
                pairs.add((i, j))
                continue

            # Heading overlap
            keywords_j = _heading_keywords(chunks[j].heading)
            if len(keywords_i & keywords_j) >= config.min_heading_overlap:
                pairs.add((i, j))
                continue

            # Domain anchor topic overlap
            if anchor_words:
                content_i = chunks[i].content.lower()
                content_j = chunks[j].content.lower()
                shared_anchors = sum(
                    1 for a in anchor_words
                    if a in content_i and a in content_j
                )
                if shared_anchors >= 2:
                    pairs.add((i, j))

    return sorted(pairs)


# =====================================================================
# Layer B — Detectors
# =====================================================================

_NUMBER_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*(business\s+days?|days?|hours?|months?|weeks?|%|percent|dollars?|euros?)', re.IGNORECASE)
_DOLLAR_RE = re.compile(r'\$(\d+(?:\.\d+)?)\s*(?:/\s*(mo|month|year|yr|day|hour))?', re.IGNORECASE)


def _extract_numeric_facts(content: str) -> list[tuple[str, str, str]]:
    """Extract (number, unit, context) tuples from content."""
    facts = []

    # Regular unit matches
    for m in _NUMBER_RE.finditer(content):
        number = m.group(1)
        unit = m.group(2).lower().strip() if m.group(2) else ""
        start = max(0, m.start() - 40)
        context = content[start:m.start()].strip()
        context_words = context.split()[-5:]
        context = " ".join(context_words).lower()
        facts.append((number, unit, context))

    # Dollar amounts
    for m in _DOLLAR_RE.finditer(content):
        number = m.group(1)
        period = m.group(2)
        unit = f"$/{period.lower()}" if period else "$"
        start = max(0, m.start() - 40)
        context = content[start:m.start()].strip()
        context_words = context.split()[-5:]
        context = " ".join(context_words).lower()
        facts.append((number, unit, context))

    return facts


def _detect_numeric_conflict(chunk_a: Chunk, chunk_b: Chunk,
                             scope_a: dict = None, scope_b: dict = None) -> Optional[ConsistencyFinding]:
    """Find numeric contradictions between two chunks.

    If both chunks have a number with the same unit in similar context,
    and the numbers differ, it's a contradiction.
    Skips if scope differs (US vs Asia, Starter vs Enterprise).
    """
    facts_a = _extract_numeric_facts(chunk_a.content)
    facts_b = _extract_numeric_facts(chunk_b.content)

    # Generic words that appear everywhere — don't count for context matching
    _generic_ctx = {
        "the", "a", "an", "of", "for", "all", "new", "our", "are", "is",
        "will", "can", "may", "from", "with", "within", "after", "before",
        "each", "per", "over", "up", "to", "on", "in", "at", "by",
        "customer", "customers", "payment", "payments", "account",
        "plan", "service", "process", "time", "period", "date",
        "total", "amount", "number", "based", "times",
    }

    for num_a, unit_a, ctx_a in facts_a:
        for num_b, unit_b, ctx_b in facts_b:
            if num_a == num_b:
                continue
            if not unit_a or unit_a != unit_b:
                continue
            # Check context overlap — filter out generic words
            ctx_words_a = set(ctx_a.split()) - _generic_ctx
            ctx_words_b = set(ctx_b.split()) - _generic_ctx
            overlap = ctx_words_a & ctx_words_b
            # Need at least 2 specific words in common (not generic)
            if len(overlap) < 2:
                continue

            # Check scope difference (if both have scope info)
            if scope_a and scope_b and scope_a != scope_b:
                continue  # different scope = not a conflict

            # Find the evidence sentences
            evidence_a = _find_sentence_with(chunk_a.content, num_a, unit_a)
            evidence_b = _find_sentence_with(chunk_b.content, num_b, unit_b)

            return ConsistencyFinding(
                finding_id=f"{chunk_a.chunk_id}__{chunk_b.chunk_id}_numeric",
                chunk_a_id=chunk_a.chunk_id,
                chunk_b_id=chunk_b.chunk_id,
                conflict_type="numeric",
                evidence_a=evidence_a,
                evidence_b=evidence_b,
                rationale=f'"{num_a} {unit_a}" vs "{num_b} {unit_b}" for similar context',
                suggested_winner="",  # set by date-aware suggestion
                suggestion_reason="",
            )
    return None


_AUTHORITY_VERB_RE = None  # built lazily from DomainContext


def _extract_authority_claims(content: str, domain_context: DomainContext) -> list:
    """Extract (actor, responsibility) tuples from content."""
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(a) for a in domain_context.actors) + r')\s+'
        r'(?:handles?|manages?|processes?|approves?|owns?|reviews?)\s+'
        r'(?:all\s+)?([a-z\s]+?)(?:\.|,|;|$)',
        re.IGNORECASE,
    )
    return [(m.group(1).lower(), m.group(2).lower().strip()) for m in pattern.finditer(content)]


def _detect_authority_conflict(chunk_a: Chunk, chunk_b: Chunk,
                               domain_context: Optional[DomainContext]) -> Optional[ConsistencyFinding]:
    """Detect when two chunks assign same responsibility to different actors.

    Also catches within-chunk conflicts when chunk_a == chunk_b is called separately.
    """
    if not domain_context or not domain_context.actors:
        return None

    claims_a = _extract_authority_claims(chunk_a.content, domain_context)
    claims_b = _extract_authority_claims(chunk_b.content, domain_context)

    for actor_a, responsibility_a in claims_a:
        for actor_b, responsibility_b in claims_b:
            if actor_a == actor_b:
                continue
            resp_words_a = set(responsibility_a.split())
            resp_words_b = set(responsibility_b.split())
            if len(resp_words_a & resp_words_b) >= 2:
                return ConsistencyFinding(
                    finding_id=f"{chunk_a.chunk_id}__{chunk_b.chunk_id}_authority",
                    chunk_a_id=chunk_a.chunk_id,
                    chunk_b_id=chunk_b.chunk_id,
                    conflict_type="authority",
                    evidence_a=_find_sentence_with(chunk_a.content, actor_a, ""),
                    evidence_b=_find_sentence_with(chunk_b.content, actor_b, ""),
                    rationale=f'"{actor_a}" vs "{actor_b}" assigned same responsibility',
                    suggested_winner="",
                    suggestion_reason="",
                )
    return None


def _detect_authority_within_chunk(chunk: Chunk,
                                   domain_context: Optional[DomainContext]) -> Optional[ConsistencyFinding]:
    """Detect authority conflict within a single chunk."""
    if not domain_context or not domain_context.actors:
        return None

    claims = _extract_authority_claims(chunk.content, domain_context)
    if len(claims) < 2:
        return None

    # Check if multiple actors claim same responsibility within this chunk
    for i, (actor_a, resp_a) in enumerate(claims):
        for j, (actor_b, resp_b) in enumerate(claims):
            if i >= j or actor_a == actor_b:
                continue
            resp_words_a = set(resp_a.split())
            resp_words_b = set(resp_b.split())
            if len(resp_words_a & resp_words_b) >= 2:
                return ConsistencyFinding(
                    finding_id=f"{chunk.chunk_id}_authority_internal",
                    chunk_a_id=chunk.chunk_id,
                    chunk_b_id=chunk.chunk_id,
                    conflict_type="authority",
                    evidence_a=_find_sentence_with(chunk.content, actor_a, ""),
                    evidence_b=_find_sentence_with(chunk.content, actor_b, ""),
                    rationale=f'Within same chunk: "{actor_a}" vs "{actor_b}" claim same responsibility',
                    suggested_winner="",
                    suggestion_reason="Cannot auto-suggest — same chunk",
                )
    return None


def _detect_process_conflict(chunk_a: Chunk, chunk_b: Chunk) -> Optional[ConsistencyFinding]:
    """Detect different step sequences for the same task.

    Looks for numbered steps or "Step N:" patterns. If two chunks describe
    the same task with different step content, flag it.
    """
    steps_a = re.findall(r'(?:^|\s)(?:\d+[.)]\s+|Step \d+:\s*)([^.!?\n]+)', chunk_a.content)
    steps_b = re.findall(r'(?:^|\s)(?:\d+[.)]\s+|Step \d+:\s*)([^.!?\n]+)', chunk_b.content)

    if len(steps_a) >= 2 and len(steps_b) >= 2:
        # Check if first steps differ significantly
        first_a = steps_a[0].lower().split()[:5]
        first_b = steps_b[0].lower().split()[:5]
        set_a = set(first_a)
        set_b = set(first_b)
        overlap = len(set_a & set_b)
        if overlap <= 1:  # very little overlap = different first step
            # Check if headings suggest same task
            h_a = _heading_keywords(chunk_a.heading)
            h_b = _heading_keywords(chunk_b.heading)
            if h_a & h_b:
                return ConsistencyFinding(
                    finding_id=f"{chunk_a.chunk_id}__{chunk_b.chunk_id}_process",
                    chunk_a_id=chunk_a.chunk_id,
                    chunk_b_id=chunk_b.chunk_id,
                    conflict_type="process",
                    evidence_a=f"{len(steps_a)} steps starting: {steps_a[0][:60]}",
                    evidence_b=f"{len(steps_b)} steps starting: {steps_b[0][:60]}",
                    rationale="Same topic, different procedure steps",
                    suggested_winner="",
                    suggestion_reason="",
                )
    return None


# =====================================================================
# Scope filtering (from A21 scope_qualifiers)
# =====================================================================

def _extract_chunk_scope(chunk: Chunk, domain_context: Optional[DomainContext]) -> dict:
    """Extract scope qualifiers present in a chunk."""
    if not domain_context:
        return {}

    scope = {}
    content_lower = chunk.content.lower()
    heading_lower = chunk.heading.lower()
    full = content_lower + " " + heading_lower

    for sq in domain_context.scope_qualifiers:
        stype = sq["type"]
        for val in sq["values"]:
            if val.lower() in full:
                scope[stype] = val.lower()
                break
    return scope


# =====================================================================
# Date-aware suggestion
# =====================================================================

def _suggest_winner_by_date(finding: ConsistencyFinding,
                            chunk_a: Chunk, chunk_b: Chunk) -> tuple[str, str]:
    """Suggest the newer chunk as winner based on A22 metadata.

    Returns (winner, reason): winner is "a", "b", or "" if no date info.
    """
    age_a = chunk_a.metadata.get("a22_page_age_months")
    age_b = chunk_b.metadata.get("a22_page_age_months")

    # Page metadata wins if available
    if age_a is not None and age_b is not None:
        if age_a < age_b:
            return "a", f"Chunk A page is newer ({age_a}m vs {age_b}m old)"
        elif age_b < age_a:
            return "b", f"Chunk B page is newer ({age_b}m vs {age_a}m old)"

    # Fall back to content dates
    date_a = chunk_a.metadata.get("a22_newest_content_date")
    date_b = chunk_b.metadata.get("a22_newest_content_date")

    if date_a and date_b:
        age_a_content = date_a.get("age_months", 999)
        age_b_content = date_b.get("age_months", 999)
        if age_a_content < age_b_content:
            return "a", f"Chunk A mentions newer date ({date_a['raw']} vs {date_b['raw']})"
        elif age_b_content < age_a_content:
            return "b", f"Chunk B mentions newer date ({date_b['raw']} vs {date_a['raw']})"

    return "", ""


# =====================================================================
# Helpers
# =====================================================================

def _find_sentence_with(text: str, token: str, token2: str = "") -> str:
    """Find the sentence containing a token."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if token.lower() in sent.lower():
            if not token2 or token2.lower() in sent.lower():
                return sent.strip()[:200]
    return text[:200]


# =====================================================================
# Main checker
# =====================================================================

def _llm_validate_contradictions(findings: list[ConsistencyFinding],
                                  chunks: list[Chunk],
                                  llm_call, domain_type: str) -> list[ConsistencyFinding]:
    """LLM validates contradictions — removes false positives, suggests winners."""
    import json as _json

    if not findings:
        return findings

    chunk_map = {c.chunk_id: c for c in chunks}

    items = []
    for i, f in enumerate(findings):
        chunk_a = chunk_map.get(f.chunk_a_id)
        chunk_b = chunk_map.get(f.chunk_b_id)
        heading_a = chunk_a.heading[:40] if chunk_a else f.chunk_a_id
        heading_b = chunk_b.heading[:40] if chunk_b else f.chunk_b_id

        items.append(
            f'{i+1}. [{f.conflict_type}] {f.rationale}\n'
            f'   A ({f.chunk_a_id}, "{heading_a}"): {f.evidence_a[:120]}\n'
            f'   B ({f.chunk_b_id}, "{heading_b}"): {f.evidence_b[:120]}'
        )
    items_text = "\n\n".join(items)

    prompt = f"""You are validating contradictions found in a {domain_type} knowledge base.
A rule-based scanner found these potential contradictions. Some may be false positives.

CONTRADICTIONS FOUND:
{items_text}

For each contradiction, determine:
1. Is this a REAL contradiction or a FALSE POSITIVE?
2. If real: which side (A or B) is more likely correct and should be kept? Why?
3. How confident are you?

VALIDATION RULES:
- Different topics using similar numbers = FALSE POSITIVE ("7 day retry period" vs "14 day trial" are different things)
- Same topic with different numbers = REAL ("refund takes 5 days" vs "refund takes 10 days")
- Same task with different procedures = REAL (two different ways to do the same thing)
- Same responsibility claimed by different teams = REAL
- Scoped differences (US vs Europe, Starter vs Enterprise) = NOT a contradiction
- If both sides could be true in different contexts, prefer "keep_both" with explanation

WINNER SELECTION:
- If one side is more specific or detailed → prefer it
- If one side uses more recent language/dates → prefer it
- If one side is the official process (from a dedicated section) vs a casual mention → prefer the dedicated section
- If genuinely uncertain → suggest "keep_both"

Return as JSON array:
[
  {{"finding": 1, "valid": true, "suggested_winner": "a|b|keep_both",
    "reason": "why this winner", "confidence": "high|medium|low"}}
]

Return ONLY the JSON array."""

    try:
        result = llm_call(prompt)
        if not result:
            return findings

        text = result.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return findings

        validations = _json.loads(json_match.group())

        validated = []
        mentioned = set()

        for v in validations:
            idx = v.get("finding", 0) - 1
            if idx < 0 or idx >= len(findings):
                continue
            mentioned.add(idx)
            f = findings[idx]

            if not v.get("valid", True):
                continue  # false positive — drop

            # Update with LLM suggestion
            winner = v.get("suggested_winner", "")
            reason = v.get("reason", "")
            confidence = v.get("confidence", "medium")

            if winner in ("a", "b"):
                f.suggested_winner = winner
                f.suggestion_reason = reason
            elif winner == "keep_both":
                f.suggested_winner = ""
                f.suggestion_reason = f"LLM: keep both — {reason}"

            f.confidence = confidence

            # Smart default: high confidence → pre-set user_decision
            if confidence == "high" and winner in ("a", "b"):
                f.user_decision = f"select_{winner}"

            validated.append(f)

        # Keep unmentioned findings
        for i, f in enumerate(findings):
            if i not in mentioned:
                validated.append(f)

        return validated

    except Exception as e:
        import logging
        logging.getLogger("aiq.a32").warning("LLM validation failed: %s", e)
        return findings


class ConsistencyChecker:
    """A32 — Find contradictions between chunks."""

    def __init__(self, config: Optional[A32Config] = None):
        self.config = config or A32Config()

    def run(self, chunks: list[Chunk],
            domain_context: Optional[DomainContext] = None) -> ModuleOutput:
        """Find contradictions across all chunk pairs.

        Args:
            chunks: from A14 (enriched by A21/A22, classified by A31)
            domain_context: from A21

        Returns:
            ModuleOutput with consistency findings
        """
        t0 = time.perf_counter()
        words_in = sum(c.words for c in chunks)

        # Layer A: candidate pairs
        pairs = _build_candidate_pairs(chunks, self.config, domain_context)

        # Pre-compute scope for each chunk
        chunk_scope = {c.chunk_id: _extract_chunk_scope(c, domain_context) for c in chunks}

        findings: list[ConsistencyFinding] = []

        # Within-chunk authority conflicts (same chunk mentions multiple actors with same responsibility)
        for chunk in chunks:
            if chunk.tag.value in ("placeholder", "editorial"):
                continue
            internal_finding = _detect_authority_within_chunk(chunk, domain_context)
            if internal_finding:
                findings.append(internal_finding)

        # Layer B: run detectors on each pair
        for i, j in pairs:
            chunk_a = chunks[i]
            chunk_b = chunks[j]
            scope_a = chunk_scope.get(chunk_a.chunk_id, {})
            scope_b = chunk_scope.get(chunk_b.chunk_id, {})

            # Numeric conflict
            finding = _detect_numeric_conflict(chunk_a, chunk_b, scope_a, scope_b)
            if finding:
                winner, reason = _suggest_winner_by_date(finding, chunk_a, chunk_b)
                finding.suggested_winner = winner
                finding.suggestion_reason = reason
                findings.append(finding)
                continue  # one finding per pair is enough

            # Authority conflict
            finding = _detect_authority_conflict(chunk_a, chunk_b, domain_context)
            if finding:
                winner, reason = _suggest_winner_by_date(finding, chunk_a, chunk_b)
                finding.suggested_winner = winner
                finding.suggestion_reason = reason
                findings.append(finding)
                continue

            # Process conflict
            finding = _detect_process_conflict(chunk_a, chunk_b)
            if finding:
                winner, reason = _suggest_winner_by_date(finding, chunk_a, chunk_b)
                finding.suggested_winner = winner
                finding.suggestion_reason = reason
                findings.append(finding)
                continue

        # Layer C: LLM judge (optional, not implemented in rule-based version)
        # LLM validation — confirm/reject, suggest winner, set confidence
        if self.config.llm_call and findings:
            findings = _llm_validate_contradictions(
                findings, chunks, self.config.llm_call,
                domain_context.domain_type if domain_context else "general")

        detected = len(findings)

        return ModuleOutput(
            module_id="A32",
            module_name="Consistency",
            detected=detected,
            resolved=0,  # resolved count comes from user decisions later
            remaining=detected,
            words_in=words_in,
            words_out=words_in,
            findings=findings,
            elapsed_seconds=time.perf_counter() - t0,
            data={
                "pairs_evaluated": len(pairs),
                "total_chunks": len(chunks),
            },
        )

    def apply_user_decisions(self, chunks: list[Chunk], findings: list[ConsistencyFinding],
                             bulk_accept: bool = False) -> dict:
        """Apply user decisions to chunks.

        Args:
            chunks: list of chunks
            findings: with user_decision set
            bulk_accept: if True, "accept all" was clicked — use suggested_winner for pending

        Returns:
            dict with counts: selected_a, selected_b, keep_both, caveat, unresolved
        """
        from aiq.core.types import ChunkTag

        chunk_map = {c.chunk_id: c for c in chunks}
        stats = {"selected_a": 0, "selected_b": 0, "keep_both": 0, "caveat": 0, "unresolved": 0}

        for f in findings:
            decision = f.user_decision

            # Bulk accept: use suggested winner if still pending
            if bulk_accept and decision == "pending" and f.suggested_winner:
                decision = f"select_{f.suggested_winner}"
                f.user_decision = decision

            if decision == "select_a":
                # Tag B as superseded
                if f.chunk_b_id in chunk_map:
                    chunk_map[f.chunk_b_id].tag = ChunkTag.STALE
                    chunk_map[f.chunk_b_id].tag_reason = f"Superseded by {f.chunk_a_id} (user decision)"
                    chunk_map[f.chunk_b_id].tag_module = "A32"
                stats["selected_a"] += 1
            elif decision == "select_b":
                if f.chunk_a_id in chunk_map:
                    chunk_map[f.chunk_a_id].tag = ChunkTag.STALE
                    chunk_map[f.chunk_a_id].tag_reason = f"Superseded by {f.chunk_b_id} (user decision)"
                    chunk_map[f.chunk_a_id].tag_module = "A32"
                stats["selected_b"] += 1
            elif decision == "keep_both":
                # No tag change — saved as "not a conflict" for future feedback loop
                stats["keep_both"] += 1
            elif decision == "pending":
                # Unresolved — attach conflict caveat to both chunks so retrieval
                # can warn the end user that conflicting information exists.
                self._attach_conflict_caveat(f, chunk_map)
                stats["caveat"] += 1
            else:
                stats["unresolved"] += 1

        stats["resolved"] = stats["selected_a"] + stats["selected_b"] + stats["keep_both"] + stats["caveat"]
        return stats

    @staticmethod
    def _attach_conflict_caveat(finding: ConsistencyFinding, chunk_map: dict):
        """Store conflict info on both chunks so retrieval surfaces a caveat.

        When chunk A is retrieved, the caveat warns: "Note: this may conflict
        with [chunk B heading] which states [evidence_b]." And vice versa.
        """
        chunk_a = chunk_map.get(finding.chunk_a_id)
        chunk_b = chunk_map.get(finding.chunk_b_id)

        if chunk_a:
            conflicts = chunk_a.metadata.setdefault("a32_conflicts", [])
            conflicts.append({
                "conflict_type": finding.conflict_type,
                "conflicts_with": finding.chunk_b_id,
                "conflicts_with_heading": chunk_b.heading if chunk_b else "",
                "their_evidence": finding.evidence_b[:200],
                "rationale": finding.rationale,
            })

        if chunk_b:
            conflicts = chunk_b.metadata.setdefault("a32_conflicts", [])
            conflicts.append({
                "conflict_type": finding.conflict_type,
                "conflicts_with": finding.chunk_a_id,
                "conflicts_with_heading": chunk_a.heading if chunk_a else "",
                "their_evidence": finding.evidence_a[:200],
                "rationale": finding.rationale,
            })
