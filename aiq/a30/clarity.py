"""A30 — Semantic Clarity

Detects and optionally fixes ambiguous content that would confuse
retrieval systems or end users reading the retrieved chunks.

What it detects (6 issue types):
    1. Dangling pronouns — "They handle escalations" (who is "they"?)
    2. Undefined acronyms — "Check the SLA" (not expanded in this chunk)
    3. Vague entity references — "The team processes refunds" (which team?)
    4. Broken cross-references — "As mentioned above" (no "above" in this chunk)
    5. Complex sentences — 40+ words, hard to parse
    6. Sequence gaps — Step 1, Step 3 (missing Step 2)

How it works:
    - Detection is always rule-based (regex patterns)
    - Each finding gets a confidence level (confident/not confident)
    - Fixing has three modes per issue type:
      detect_only: flag only, don't change content
      rule_fix:    auto-fix using DomainContext (acronym expansion, pronoun resolution)
      llm_fix:     LLM rewrites for clarity (two focused prompts: entity resolution + rewrite)
    - Uses A11 DomainContext for resolution:
      acronyms dict for expansion, actors dict for "the team", company_name for "the platform"

Config:
    pronoun_mode: "detect_only" | "rule_fix" | "llm_fix" (default: "detect_only")
    acronym_mode: "detect_only" | "rule_fix" | "llm_fix" (default: "rule_fix")
    vague_entity_mode: "detect_only" | "rule_fix" | "llm_fix" (default: "detect_only")
    reference_mode: "detect_only" | "llm_fix" (default: "detect_only")
    procedure_mode: "detect_only" (default: "detect_only")
    sentence_mode: "detect_only" | "llm_fix" (default: "detect_only")
    sequence_mode: "detect_only" (default: "detect_only")
    llm_call: optional callable(prompt: str) -> str
    max_sentence_words: word threshold for complex sentence detection (default: 40)

Config exposed to AIQConfig:
    clarity_pronoun_mode  -> A30Config.pronoun_mode        (default: "rule_fix")
    clarity_acronym_mode  -> A30Config.acronym_mode        (default: "rule_fix")
    max_sentence_words    -> A30Config.max_sentence_words  (default: 40)
    detection_confidence  -> filters which findings get acted on (future: numeric scoring)
    (llm_call wired from AIQConfig.llm_client)

Auto-detected (no user input needed):
    All 6 issue types detected from content via regex
    Pronoun resolution inferred from heading + DomainContext actors
    Acronym expansion from A11 acronym dictionary

    Each finding has a confidence flag:
    - confident=True: clear single answer (heading matches actor, acronym has known expansion)
    - confident=False: ambiguous (multiple actors possible, no expansion known)

    Future: numeric confidence_score (0.0-1.0) for threshold-based filtering.

Input:  list[Chunk], optional DomainContext
Output: ModuleOutput with .findings = list[ClarityFinding], .chunks = modified chunks

LLM required: No (detect_only and rule_fix work without LLM).
    LLM enhances: entity resolution for ambiguous pronouns, sentence rewrites.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import Chunk, DomainContext, ModuleOutput, TokenChange


# =====================================================================
# Config
# =====================================================================

@dataclass
class A30Config:
    """Configuration for semantic clarity.

    Fix modes per issue type:
      detect_only: flag only
      rule_fix:    auto-fix using DomainContext
      llm_fix:     LLM rewrites
    """
    pronoun_mode: str = "detect_only"
    acronym_mode: str = "rule_fix"      # rule_fix works well here (A21 has expansions)
    vague_entity_mode: str = "detect_only"
    reference_mode: str = "detect_only"
    procedure_mode: str = "detect_only"
    sentence_mode: str = "detect_only"
    sequence_mode: str = "detect_only"
    # LLM function for llm_fix modes
    llm_call: Optional[Callable] = None
    # Thresholds
    max_sentence_words: int = 40


# =====================================================================
# Finding
# =====================================================================

@dataclass
class ClarityFinding:
    """One clarity issue found in a chunk."""
    chunk_id: str
    issue_type: str         # pronoun, acronym, vague_entity, reference, procedure, sentence, sequence
    evidence: str           # the text that triggered detection
    reason: str             # human-readable explanation
    fixed: bool = False     # was this issue auto-fixed?
    fix_detail: str = ""    # what was changed
    # Proposal fields
    original_term: str = ""     # the ambiguous term ("They", "CRM", "the system")
    proposed_fix: str = ""      # our best suggestion (empty if not confident)
    proposal_source: str = ""   # why we think so
    confident: bool = False     # True = we have a clear single answer


# =====================================================================
# Detectors
# =====================================================================

_PRONOUN_RE = re.compile(
    r'(?:^|[.!?]\s+)((?:They|Them|Their|It|This|That|These|Those)\s)',
    re.MULTILINE,
)

_PRONOUN_MID_RE = re.compile(
    r'[^.!?]*\b(they|them|their)\b[^.!?]*[.!?]',
    re.IGNORECASE,
)

_REFERENCE_RE = re.compile(
    r'(?:as (?:above|mentioned|described|noted)|see previous|the above|'
    r'refer(?:red)? to (?:earlier|above|the previous)|mentioned earlier)',
    re.IGNORECASE,
)

_VAGUE_ENTITY_RE = re.compile(
    r'\b(?:the system|the tool|the platform|the service|the application|'
    r'the team|the department|the group)\b',
    re.IGNORECASE,
)


def _detect_pronouns(content: str, chunk_id: str,
                     heading: str = "",
                     domain_context: Optional[DomainContext] = None) -> list[ClarityFinding]:
    """Detect ambiguous pronouns at sentence start and mid-sentence they/them/their."""
    findings = []
    seen = set()

    proposed, source, confident = _infer_pronoun_subject(heading, domain_context)

    # Sentence-start pronouns
    for m in _PRONOUN_RE.finditer(content):
        pronoun = m.group(1).strip()
        if pronoun.lower() not in seen:
            seen.add(pronoun.lower())
            findings.append(ClarityFinding(
                chunk_id=chunk_id, issue_type="pronoun",
                evidence=_get_sentence(content, m.start(), m.end()),
                reason=f'Ambiguous pronoun "{pronoun}" — who/what does it refer to?',
                original_term=pronoun,
                proposed_fix=proposed if confident else "",
                proposal_source=source,
                confident=confident,
            ))

    # Mid-sentence they/them/their
    for m in _PRONOUN_MID_RE.finditer(content):
        pronoun = m.group(1).lower()
        if pronoun not in seen:
            seen.add(pronoun)
            findings.append(ClarityFinding(
                chunk_id=chunk_id, issue_type="pronoun",
                evidence=m.group().strip()[:100],
                reason=f'Ambiguous pronoun "{pronoun}" — who does this refer to?',
                original_term=pronoun,
                proposed_fix=proposed if confident else "",
                proposal_source=source,
                confident=confident,
            ))

    return findings


def _infer_pronoun_subject(heading: str,
                           domain_context: Optional[DomainContext] = None) -> tuple[str, str, bool]:
    """Infer likely pronoun subject from heading and domain context.

    Returns (proposed_fix, proposal_source, confident).
    Confident only when heading clearly matches an actor or product.
    """
    if not heading:
        return "", "", False

    heading_lower = heading.lower()

    # Check if heading matches a specific actor from A21
    if domain_context and domain_context.actors:
        for actor in domain_context.actors:
            if actor.lower() in heading_lower:
                return (f"the {actor} team",
                        f'heading "{heading}" mentions "{actor}"',
                        True)

    # Check product names in heading
    if domain_context and domain_context.product_names:
        for product in domain_context.product_names:
            if product.lower() in heading_lower:
                return product, f'heading "{heading}" mentions "{product}"', True

    # Fall back to heading — not confident (could be wrong)
    return heading, f'inferred from heading "{heading}"', False


def _detect_acronyms(content: str, chunk_id: str,
                     domain_context: Optional[DomainContext] = None) -> list[ClarityFinding]:
    """Detect undefined acronyms not expanded in this chunk."""
    acronym_re = re.compile(r'\b([A-Z]{2,5})\b')
    findings = []

    # Get known acronyms from domain context
    known = set()
    expansions = {}
    if domain_context:
        known = domain_context.common_acronyms.copy()
        expansions = dict(domain_context.acronyms)
        # Acronyms with expansions are "known" — but we still flag if not expanded in text
        known.update(k for k, v in expansions.items() if not v)  # truly unknown ones

    # False positives to skip — common words, error codes, Jira keys, placeholders
    false_acr = {
        "FOR", "AND", "THE", "NOT", "ALL", "USE", "SET", "GET",
        "CASE", "NOTE", "ONLY", "ALSO", "MUST", "WILL", "WHEN",
        "WITH", "FROM", "THAT", "THIS", "HAVE", "BEEN", "EACH",
        "DRAFT", "CHANGE", "LEVEL", "STEP", "CALL", "PLAN",
        # Placeholders — handled by A31 governance
        "TODO", "TBD", "FIXME", "HACK", "XXX",
        # Error code prefixes — not real acronyms
        "ERR",
        # Project tracking keys — handled by A31 metadata leak
        "JIRA", "PAY",
        # Time zones — commonly understood
        "EST", "PST", "UTC", "GMT",
        # Other common non-acronyms
        "ETA",
    }

    found = set()
    for m in acronym_re.finditer(content):
        acr = m.group(1)
        if acr in found or acr in known or acr in false_acr:
            continue

        # Check if expansion exists in content already: "CRM (Customer Relationship Management)"
        expansion_in_text = bool(re.search(
            rf'{acr}\s*\([A-Z][a-z]', content
        ))
        if expansion_in_text:
            continue

        found.add(acr)
        expansion = expansions.get(acr, "")
        findings.append(ClarityFinding(
            chunk_id=chunk_id, issue_type="acronym",
            evidence=_get_sentence(content, m.start(), m.end()),
            reason=f'Acronym "{acr}" not expanded'
                   + (f' (known: {expansion})' if expansion else ' (undefined)'),
            original_term=acr,
            proposed_fix=f"{acr} ({expansion})" if expansion else "",
            proposal_source="A21 acronym dictionary" if expansion else "",
            confident=bool(expansion),
        ))

    return findings


def _detect_vague_entities(content: str, chunk_id: str,
                           domain_context: Optional[DomainContext] = None) -> list[ClarityFinding]:
    """Detect vague entity references like 'the system', 'the team'."""
    findings = []
    seen = set()

    for m in _VAGUE_ENTITY_RE.finditer(content):
        entity = m.group().lower()
        if entity in seen:
            continue
        seen.add(entity)

        # Build proposal — use company_name for system/platform, actors for team
        proposed = ""
        source = ""
        confident = False
        if domain_context:
            if "system" in entity or "platform" in entity or "tool" in entity or "service" in entity or "application" in entity:
                # "the platform" / "the system" → company name
                if domain_context.company_name:
                    proposed = domain_context.company_name
                    source = f"A11: company name is {domain_context.company_name}"
                    confident = True
                elif domain_context.product_names and len(domain_context.product_names) == 1:
                    proposed = domain_context.product_names[0]
                    source = "A11: only known product"
                    confident = True
            elif "team" in entity or "department" in entity or "group" in entity:
                if domain_context.actors:
                    actors_list = list(domain_context.actors.keys())
                    if len(actors_list) == 1:
                        proposed = f"the {actors_list[0]} team"
                        source = "A11: only known actor"
                        confident = True
                    # Multiple actors = LLM will resolve from context

        findings.append(ClarityFinding(
            chunk_id=chunk_id, issue_type="vague_entity",
            evidence=_get_sentence(content, m.start(), m.end()),
            reason=f'Vague reference "{m.group()}" — which {entity.split()[-1]}?',
            original_term=m.group(),
            proposed_fix=proposed if confident else "",
            proposal_source=source,
            confident=confident,
        ))

    return findings


def _detect_references(content: str, chunk_id: str) -> list[ClarityFinding]:
    """Detect unresolved internal references and propose rewrites."""
    findings = []
    for m in _REFERENCE_RE.finditer(content):
        sentence = _get_sentence(content, m.start(), m.end())
        rewrite = _propose_reference_rewrite(sentence, m.group())
        # Confident if we could produce a clean rewrite
        is_confident = rewrite != sentence and "(remove" not in rewrite
        findings.append(ClarityFinding(
            chunk_id=chunk_id, issue_type="reference",
            evidence=sentence,
            reason=f'Unresolved reference: "{m.group()}" — chunk must be self-contained',
            original_term=sentence,
            proposed_fix=rewrite if is_confident else "",
            proposal_source="rewrite: removed dangling reference" if is_confident else "needs manual rewrite",
            confident=is_confident,
        ))
    return findings


def _propose_reference_rewrite(sentence: str, reference: str) -> str:
    """Propose a rewritten sentence that removes the dangling reference.

    Simple rule-based: remove the reference clause and clean up.
    """
    # Common patterns: "As mentioned above, X does Y" → "X does Y"
    # "See previous section for details" → remove entirely
    # "As shown in Figure 4, the process has 5 stages" → "The process has 5 stages"

    rewrite = sentence

    # "As shown/mentioned/described in X, ..." → remove prefix
    prefix_re = re.compile(
        r'^(?:As (?:shown|mentioned|described|noted|discussed) (?:in (?:Figure|Table|Section|the)?\s*\w*\s*,?\s*))',
        re.IGNORECASE,
    )
    m = prefix_re.match(rewrite)
    if m:
        rewrite = rewrite[m.end():]
        # Capitalize first letter
        if rewrite:
            rewrite = rewrite[0].upper() + rewrite[1:]
        return rewrite

    # "refer to X" at the end → remove
    suffix_re = re.compile(
        r'[,;]\s*(?:refer(?:red)? to (?:the |our )?(?:above|previous|earlier).*|see (?:above|previous).*)$',
        re.IGNORECASE,
    )
    rewrite = suffix_re.sub('.', rewrite)

    # If the whole sentence is just a reference → mark for removal
    if reference.lower() == sentence.strip().rstrip('.').lower():
        return "(remove this sentence — dangling reference)"

    return rewrite


def _detect_procedures(content: str, chunk_id: str) -> list[ClarityFinding]:
    """Detect incomplete procedures — numbered steps with gaps or missing verification."""
    findings = []

    # Check for numbered steps
    step_numbers = re.findall(r'^(\d+)[.)]\s', content, re.MULTILINE)
    if step_numbers:
        nums = [int(n) for n in step_numbers]
        # Sequence gap check
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                findings.append(ClarityFinding(
                    chunk_id=chunk_id, issue_type="sequence",
                    evidence=f"Steps go {nums[i-1]} to {nums[i]}",
                    reason=f"Sequence gap: step {nums[i-1]} to {nums[i]} (missing {nums[i-1]+1})",
                ))
                break

    # Check for "Step N:" patterns
    step_pattern = re.findall(r'Step (\d+):', content)
    if step_pattern:
        nums = [int(n) for n in step_pattern]
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                findings.append(ClarityFinding(
                    chunk_id=chunk_id, issue_type="sequence",
                    evidence=f"Steps go {nums[i-1]} to {nums[i]}",
                    reason=f"Step sequence gap: {nums[i-1]} to {nums[i]}",
                ))
                break

    return findings


def _detect_complex_sentences(content: str, chunk_id: str,
                              max_words: int = 40) -> list[ClarityFinding]:
    """Detect overly complex sentences."""
    findings = []
    sentences = re.split(r'(?<=[.!?])\s+', content)

    for sent in sentences:
        wc = len(sent.split())
        if wc > max_words:
            findings.append(ClarityFinding(
                chunk_id=chunk_id, issue_type="sentence",
                evidence=sent,
                reason=f"Complex sentence ({wc} words, max: {max_words}) — consider splitting",
                original_term=sent,
                confident=False,
            ))
            break  # Only flag the worst one per chunk

    return findings


# =====================================================================
# Fixers
# =====================================================================

def _fix_acronyms_rule(content: str, findings: list[ClarityFinding],
                       domain_context: Optional[DomainContext]) -> tuple[str, list[ClarityFinding], int]:
    """Expand acronyms using DomainContext expansions. Returns (new_content, updated_findings, tokens_added)."""
    if not domain_context:
        return content, findings, 0

    new_content = content
    tokens_added = 0

    for finding in findings:
        if finding.issue_type != "acronym":
            continue

        # Extract the acronym from the evidence
        acr_match = re.search(r'"([A-Z]{2,5})"', finding.reason)
        if not acr_match:
            continue
        acr = acr_match.group(1)
        expansion = domain_context.acronyms.get(acr, "")

        if expansion:
            # Replace first occurrence: "CRM" → "CRM (Customer Relationship Management)"
            pattern = rf'\b{acr}\b'
            first_match = re.search(pattern, new_content)
            if first_match:
                expanded = f"{acr} ({expansion})"
                new_content = new_content[:first_match.start()] + expanded + new_content[first_match.end():]
                tokens_added += len(expansion.split())
                finding.fixed = True
                finding.fix_detail = f'Expanded to: {expanded}'

    return new_content, findings, tokens_added


def _fix_pronouns_rule(content: str, findings: list[ClarityFinding],
                       heading: str, domain_context: Optional[DomainContext]) -> tuple[str, list[ClarityFinding], int]:
    """Resolve pronouns using heading context. Returns (new_content, updated_findings, tokens_added)."""
    # Try to infer the subject from the heading
    subject = ""
    if heading:
        # Use the heading as the likely subject
        heading_lower = heading.lower()
        if domain_context and domain_context.actors:
            for actor in domain_context.actors:
                if actor in heading_lower:
                    subject = f"the {actor} team"
                    break
        if not subject and domain_context and domain_context.product_names:
            for product in domain_context.product_names:
                if product.lower() in heading_lower:
                    subject = product
                    break
        if not subject:
            subject = heading

    if not subject:
        return content, findings, 0

    # Only fix "They" at sentence start (safest rule-based fix)
    new_content = content
    tokens_added = 0

    for finding in findings:
        if finding.issue_type != "pronoun":
            continue
        # Only fix sentence-start "They"
        match = re.search(r'(?:^|[.!?]\s+)They\s', new_content, re.MULTILINE)
        if match:
            old = match.group()
            new = old.replace("They", subject, 1)
            new_content = new_content[:match.start()] + new + new_content[match.end():]
            tokens_added += len(subject.split()) - 1
            finding.fixed = True
            finding.fix_detail = f'Replaced "They" with "{subject}"'
            break  # Only fix first occurrence to be safe

    return new_content, findings, tokens_added


# =====================================================================
# Helpers
# =====================================================================

def _get_sentence(text: str, start: int, end: int) -> str:
    """Extract the sentence containing the match."""
    s = start
    while s > 0 and text[s - 1] not in '.!?\n':
        s -= 1
    while s < start and text[s] in ' \t\n':
        s += 1

    e = end
    while e < len(text) and text[e] not in '.!?\n':
        e += 1
    if e < len(text) and text[e] in '.!?':
        e += 1

    result = text[s:e].strip()
    return result[:150] if len(result) > 150 else result


# =====================================================================
# LLM Smart Prompts
# =====================================================================

def _llm_resolve_entities(findings: list[ClarityFinding], content: str,
                          heading: str, domain_context: Optional[DomainContext],
                          llm_call: Callable):
    """Prompt 1: Resolve pronouns and vague entities using full chunk context."""
    import json

    company = domain_context.company_name if domain_context else ""
    actors = list(domain_context.actors.keys()) if domain_context else []
    products = domain_context.product_names if domain_context else []
    domain = domain_context.domain_type if domain_context else "general"

    terms_list = []
    for f in findings:
        terms_list.append(f'- "{f.original_term}" in: "{f.evidence[:100]}"')
    terms_text = "\n".join(terms_list)

    prompt = f"""You are resolving ambiguous references in a {domain} knowledge base chunk
to make it self-contained for RAG retrieval.

CHUNK HEADING: {heading}
CHUNK CONTENT:
{content[:500]}

KNOWN CONTEXT:
- Company: {company}
- Teams: {', '.join(actors) if actors else 'unknown'}
- Products: {', '.join(products) if products else 'unknown'}

AMBIGUOUS TERMS FOUND:
{terms_text}

For each term, determine what it refers to in this specific context.

Return as JSON array:
[
  {{"term": "they", "resolves_to": "the billing team", "confidence": "high"}},
  {{"term": "the platform", "resolves_to": "Neuroloft", "confidence": "high"}}
]

CONFIDENCE RULES:
- "high": the chunk context clearly indicates who/what (e.g., heading says "Billing", so "they" = billing team)
- "medium": likely but not certain (inferred from topic, not explicit)
- "low": guessing — multiple possibilities, context doesn't clarify

Return ONLY the JSON array."""

    try:
        result = llm_call(prompt)
        if not result:
            return

        # Parse JSON
        text = result.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return
        resolved = json.loads(json_match.group())

        # Apply resolutions to findings
        for r in resolved:
            term = r.get("term", "").lower()
            resolves_to = r.get("resolves_to", "")
            confidence = r.get("confidence", "low")

            if not resolves_to:
                continue

            for f in findings:
                if f.original_term.lower() == term or term in f.original_term.lower():
                    f.proposed_fix = resolves_to
                    f.proposal_source = f"LLM resolution ({confidence})"
                    f.confident = confidence == "high"
                    break

    except Exception as e:
        import logging
        logging.getLogger("aiq.a30").warning("Entity resolution LLM failed: %s", e)


def _llm_rewrite_sentences(findings: list[ClarityFinding], heading: str,
                           domain_context: Optional[DomainContext],
                           llm_call: Callable):
    """Prompt 2: Rewrite each sentence individually for completeness."""
    domain = domain_context.domain_type if domain_context else "general"

    for f in findings:
        if not f.original_term:
            continue

        issue = "dangling reference — remove the reference, keep the meaning" \
            if f.issue_type == "reference" \
            else "too complex — split into multiple shorter sentences"

        prompt = f"""You are rewriting a sentence in a {domain} knowledge base for RAG retrieval.

CHUNK HEADING: {heading}

ISSUE: {issue}

ORIGINAL SENTENCE:
{f.original_term}

REWRITE RULES:
- Split into multiple shorter sentences (max 30 words each)
- Keep EVERY fact, number, data point, and item from the original
- Do NOT skip, truncate, or summarize any items
- If the original lists 12 months, the rewrite must list all 12 months
- If the original has 5 steps, the rewrite must have all 5 steps
- Remove dangling references ("as shown above", "see previous")

CONFIDENCE:
- Add [HIGH] if meaning fully preserved
- Add [MEDIUM] if some interpretation needed

Return ONLY the rewritten text followed by the confidence tag. Nothing else."""

        try:
            result = llm_call(prompt)
            if not result or not result.strip():
                continue

            text = result.strip()
            # Extract confidence
            confidence = "medium"
            if "[HIGH]" in text.upper():
                confidence = "high"
                text = re.sub(r'\s*\[HIGH\]\s*$', '', text, flags=re.IGNORECASE).strip()
            elif "[MEDIUM]" in text.upper():
                text = re.sub(r'\s*\[MEDIUM\]\s*$', '', text, flags=re.IGNORECASE).strip()

            if text:
                f.proposed_fix = text
                f.proposal_source = f"LLM rewrite ({confidence})"
                f.confident = confidence == "high"

        except Exception as e:
            import logging
            logging.getLogger("aiq.a30").warning("Rewrite LLM failed for %s: %s",
                                                  f.chunk_id, e)


# =====================================================================
# Main checker
# =====================================================================

class ClarityChecker:
    """A30 — Detect and optionally fix clarity issues."""

    def __init__(self, config: Optional[A30Config] = None):
        self.config = config or A30Config()

    def run(self, chunks: list[Chunk],
            domain_context: Optional[DomainContext] = None) -> ModuleOutput:
        """Check all chunks for clarity issues.

        Args:
            chunks: from A14 (possibly tagged by A31)
            domain_context: from A21

        Returns:
            ModuleOutput with findings and optionally fixed chunks
        """
        t0 = time.perf_counter()
        words_in = sum(c.words for c in chunks)
        all_findings: list[ClarityFinding] = []
        total_tokens_added = 0
        fixed_count = 0

        for chunk in chunks:
            content = chunk.content
            chunk_findings: list[ClarityFinding] = []

            # 1. Pronouns (uses heading + DomainContext for proposals)
            chunk_findings.extend(_detect_pronouns(content, chunk.chunk_id, chunk.heading, domain_context))

            # 2. Acronyms (uses DomainContext)
            chunk_findings.extend(_detect_acronyms(content, chunk.chunk_id, domain_context))

            # 3. Vague entities (uses DomainContext)
            chunk_findings.extend(_detect_vague_entities(content, chunk.chunk_id, domain_context))

            # 4. Unresolved references
            chunk_findings.extend(_detect_references(content, chunk.chunk_id))

            # 5. Procedures / sequence gaps
            chunk_findings.extend(_detect_procedures(content, chunk.chunk_id))

            # 6. Complex sentences
            chunk_findings.extend(_detect_complex_sentences(
                content, chunk.chunk_id, self.config.max_sentence_words))

            # Apply fixes based on mode
            new_content = content
            tokens_added = 0

            # Fix acronyms
            if self.config.acronym_mode == "rule_fix" and domain_context:
                new_content, chunk_findings, ta = _fix_acronyms_rule(
                    new_content, chunk_findings, domain_context)
                tokens_added += ta

            # Fix pronouns
            if self.config.pronoun_mode == "rule_fix":
                new_content, chunk_findings, ta = _fix_pronouns_rule(
                    new_content, chunk_findings, chunk.heading, domain_context)
                tokens_added += ta

            # LLM smart prompts — two focused calls per chunk
            if self.config.llm_call:
                unfixed = [f for f in chunk_findings if not f.fixed]

                # Prompt 1: Entity resolution (pronouns + vague entities)
                entity_findings = [f for f in unfixed
                                   if f.issue_type in ("pronoun", "vague_entity")]
                if entity_findings:
                    _llm_resolve_entities(
                        entity_findings, new_content, chunk.heading,
                        domain_context, self.config.llm_call)

                # Prompt 2: Rewrite (references + complex sentences)
                rewrite_findings = [f for f in unfixed
                                    if f.issue_type in ("reference", "sentence")]
                if rewrite_findings:
                    _llm_rewrite_sentences(
                        rewrite_findings, chunk.heading,
                        domain_context, self.config.llm_call)

            # Update chunk if content changed
            if new_content != content:
                chunk.content = new_content
                chunk.words = len(new_content.split())
                fixed_count += 1

                if tokens_added > 0:
                    chunk.token_changes.append(TokenChange(
                        change_type="added",
                        reason="clarity_fix",
                        token_count=tokens_added,
                        module="A30",
                        detail=f"{sum(1 for f in chunk_findings if f.fixed)} issues fixed",
                    ))

            total_tokens_added += max(0, tokens_added)
            all_findings.extend(chunk_findings)

        detected = len(all_findings)
        resolved = sum(1 for f in all_findings if f.fixed)
        words_out = sum(c.words for c in chunks)

        token_changes = []
        if total_tokens_added > 0:
            token_changes.append(TokenChange(
                change_type="added",
                reason="clarity_fix",
                token_count=total_tokens_added,
                module="A30",
                detail=f"{resolved} issues fixed across {fixed_count} chunks",
            ))

        return ModuleOutput(
            module_id="A30",
            module_name="Semantic Clarity",
            detected=detected,
            resolved=resolved,
            remaining=detected - resolved,
            words_in=words_in,
            words_out=words_out,
            findings=all_findings,
            chunks=chunks,
            token_changes=token_changes,
            elapsed_seconds=time.perf_counter() - t0,
        )
