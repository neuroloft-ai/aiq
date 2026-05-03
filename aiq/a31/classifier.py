"""A31 — Content Governance

Scans chunks and classifies content safety for retrieval. Tags chunks
(never edits/deletes content). The most severe tag wins when multiple
issues exist in one chunk.

What it detects (9 detector types):
    1. Internal notes — "INTERNAL NOTE", "do not share", "agents only"
    2. PII — emails, phone numbers, person names, financial account numbers
    3. Placeholders — TODO, TBD, FIXME, "coming soon", "[insert...]"
    4. Editorial artifacts — tracked changes, HTML comments, draft markers, strikethrough
    5. Metadata leaks — Jira IDs, internal URLs, Slack channels, build numbers
    6. Vague claims — "handles most cases", "seamlessly integrates"
    7. Destructive actions — "delete account", "purge data" (base + domain-specific from A11)
    8. Broken references — "see FAQ", "image not available"
    9. Escalation gaps — "escalate to management" without contact details

How it works:
    - 9 rule-based regex detectors, each returns findings with evidence
    - Optional LLM validation: sends all findings for a chunk in one call,
      LLM confirms/rejects each, adjusts detector type, adds missed issues
    - Tag priority: PII > internal > metadata_leak > editorial > placeholder >
      destructive > vague_claim > broken_reference > escalation > content
    - Two tag categories:
      auto-block: PII, internal, placeholder, editorial, metadata_leak (blocked from retrieval)
      user-review: vague_claim, destructive, broken_reference, escalation (served with caveat)

    Remediation (separate step):
    - Noise tags (placeholder, editorial, metadata_leak): remove flagged sentences
    - Content tags (PII, internal): LLM rewrites to redact sensitive parts
    - Review tags: left as-is for user decision

Config:
    detect_internal/pii/placeholder/editorial/metadata_leak/vague/destructive/broken_ref/escalation:
        bool — enable/disable each detector (all True by default)
    pii_mode: "strict" | "smart" | "lenient" (default: "smart")
        strict:  block ALL emails, phones, names
        smart:   skip functional emails (support@), published support numbers
        lenient: only block names paired with emails/phones
    safe_email_prefixes: list of functional email prefixes to skip in smart mode
    llm_call: optional callable for false-positive validation

Config exposed to AIQConfig:
    pii_mode              -> A31Config.pii_mode        (default: "smart")
    detection_confidence  -> filters findings before tagging (future: numeric scoring)
    (llm_call wired from AIQConfig.llm_client)
    (individual detector toggles available via A31Config for advanced users)

Auto-detected (no user input needed):
    All 9 issue types detected from content via regex
    Each finding has confidence: "high" / "medium" / "low"
    Tag assignment based on priority (most severe wins)
    Destructive patterns from A11 DomainContext

    Future: numeric confidence_score for threshold-based filtering.

Input:  list[Chunk], optional DomainContext
Output: ModuleOutput with .findings = list[ClassificationFinding], .chunks = tagged chunks

LLM required: No (all 9 detectors are rule-based).
    LLM enhances: validates findings to reduce false positives, discovers missed issues.
    LLM remediation: rewrites PII/internal chunks to make them safe.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from aiq.core.types import Chunk, ChunkTag, DomainContext, ModuleOutput, TokenChange


# =====================================================================
# Config
# =====================================================================

@dataclass
class A31Config:
    """Configuration for classification."""
    # Which detectors to run (all on by default)
    detect_internal: bool = True
    detect_pii: bool = True
    detect_placeholder: bool = True
    detect_editorial: bool = True
    detect_metadata_leak: bool = True
    detect_vague: bool = True
    detect_destructive: bool = True
    detect_broken_ref: bool = True
    detect_escalation: bool = True
    # PII sensitivity: "strict" | "smart" | "lenient"
    pii_mode: str = "smart"
    # Email prefixes considered safe (functional, not personal)
    safe_email_prefixes: list = None
    # LLM for context-aware PII validation
    llm_call: Optional[callable] = None

    def __post_init__(self):
        if self.safe_email_prefixes is None:
            self.safe_email_prefixes = [
                "support", "info", "billing", "noreply", "admin", "help", "sales",
                "dpo", "privacy", "security", "compliance", "hr", "legal",
                "feedback", "contact", "hello", "team", "office",
            ]


# =====================================================================
# Finding dataclass
# =====================================================================

@dataclass
class ClassificationFinding:
    """One classification finding on a chunk."""
    chunk_id: str
    tag: ChunkTag
    detector: str           # which detector found it
    evidence: str           # the actual text that triggered detection
    reason: str             # human-readable explanation
    confidence: str = "high"  # "high" | "medium" | "low" — from LLM validation


# =====================================================================
# Tag priority (most severe wins)
# =====================================================================

_TAG_PRIORITY = {
    ChunkTag.PII: 1,
    ChunkTag.INTERNAL_ONLY: 2,
    ChunkTag.METADATA_LEAK: 3,
    ChunkTag.EDITORIAL: 4,
    ChunkTag.PLACEHOLDER: 5,
    ChunkTag.DESTRUCTIVE: 6,
    ChunkTag.VAGUE_CLAIM: 7,
    ChunkTag.BROKEN_REFERENCE: 8,
    ChunkTag.ESCALATION: 9,
    ChunkTag.CONTENT: 99,
}


# =====================================================================
# Detectors
# =====================================================================

def _detect_internal(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect internal notes and confidential markers."""
    pattern = re.compile(
        r"(?:INTERNAL\s*(?:NOTE|ONLY|USE)|"
        r"do not share(?:\s+with\s+customers)?|"
        r"not for (?:customer|external|public)|"
        r"agents?\s+(?:only|should)|"
        r"for internal use only|"
        r"confidential|"
        r"internal knowledge base)",
        re.IGNORECASE,
    )
    findings = []
    for m in pattern.finditer(content):
        evidence = _get_sentence(content, m.start(), m.end())
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.INTERNAL_ONLY,
            detector="internal_note",
            evidence=evidence,
            reason=f'Internal marker found: "{m.group()}"',
        ))
    return findings


def _detect_pii(content: str, chunk_id: str,
                pii_mode: str = "smart",
                safe_prefixes: list = None) -> list[ClassificationFinding]:
    """Detect personally identifiable information with context awareness.

    Modes:
      strict:  block ALL emails, phones, names, account numbers
      smart:   skip functional emails, published numbers; use context for ambiguous cases
      lenient: only block personal names paired with emails/phones
    """
    if safe_prefixes is None:
        safe_prefixes = ["support", "info", "billing", "noreply", "admin", "help", "sales"]

    findings = []

    if pii_mode == "lenient":
        pass  # name detection below handles this
    else:
        # ── Email addresses ──
        email_re = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
        for m in email_re.finditer(content):
            local = m.group().split("@")[0].lower()
            if pii_mode == "smart" and local in safe_prefixes:
                continue
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.PII,
                detector="pii_email",
                evidence=m.group(),
                reason=f"Personal email address: {m.group()}",
            ))

        # ── Phone numbers (context-aware) ──
        phone_re = re.compile(
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            r'(?:\s*(?:ext|x)\.?\s*\d+)?',
        )
        for m in phone_re.finditer(content):
            # Get surrounding context to determine what this number is
            ctx_start = max(0, m.start() - 80)
            ctx_end = min(len(content), m.end() + 30)
            context = content[ctx_start:ctx_end].lower()

            if pii_mode == "smart":
                # Skip published support numbers
                if any(w in context for w in ("support", "helpdesk", "call us",
                                              "reach us", "contact us")):
                    continue

                # Skip bank/financial numbers — NOT phone numbers
                if any(w in context for w in ("account", "routing", "bank",
                                              "swift", "iban", "reference",
                                              "invoice", "transaction")):
                    # This is a financial number, not a phone
                    findings.append(ClassificationFinding(
                        chunk_id=chunk_id, tag=ChunkTag.PII,
                        detector="pii_financial",
                        evidence=_get_sentence(content, m.start(), m.end()),
                        reason=f"Financial account number: {m.group()}",
                    ))
                    continue

                # Skip case/ticket IDs
                if any(w in context for w in ("case", "ticket", "jira", "issue")):
                    continue

            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.PII,
                detector="pii_phone",
                evidence=_get_sentence(content, m.start(), m.end()),
                reason=f"Phone number: {m.group()}",
            ))

        # ── Bank/financial account numbers (not caught by phone regex) ──
        account_re = re.compile(
            r'(?:account|routing|iban|swift)[:\s]+(\d{6,})',
            re.IGNORECASE,
        )
        for m in account_re.finditer(content):
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.PII,
                detector="pii_financial",
                evidence=_get_sentence(content, m.start(), m.end()),
                reason=f"Financial account number: {m.group(1)}",
            ))

    # ── Person names ──
    _GENERIC_WORDS = {
        "the", "our", "your", "their", "card", "account", "finance",
        "support", "billing", "sales", "on", "team", "management",
        "appropriate", "relevant", "issuer", "manager",
    }

    name_label_re = re.compile(
        r'(?:Contact|Author|Edited by|Assigned to|Lead|Customer|Client)[:\s]+([A-Z][a-z]{2,} [A-Z][a-z]{2,})',
    )
    for m in name_label_re.finditer(content):
        name = m.group(1)
        name_words = {w.lower() for w in name.split()}
        if not name_words & _GENERIC_WORDS:
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.PII,
                detector="pii_name",
                evidence=_get_sentence(content, m.start(), m.end()),
                reason=f"Person name: {name}",
            ))

    name_email_re = re.compile(
        r'([A-Z][a-z]{2,} [A-Z][a-z]{2,})\s*\([a-zA-Z0-9._%+-]+@',
    )
    for m in name_email_re.finditer(content):
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.PII,
            detector="pii_name_email",
            evidence=_get_sentence(content, m.start(), m.end()),
            reason=f"Person name with email: {m.group(1)}",
        ))

    return findings


def _detect_placeholder(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect placeholder and incomplete content. Deduplicates per sentence."""
    pattern = re.compile(
        r"(?:TODO\b|TBD\b|FIXME\b|HACK\b|XXX\b|"
        r"coming soon|placeholder|"
        r"to be (?:determined|added|defined|completed|updated)|"
        r"add (?:information|details|section|content) (?:about|for|here)|"
        r"\[insert .+?\])",
        re.IGNORECASE,
    )
    findings = []
    seen_sentences = set()
    for m in pattern.finditer(content):
        evidence = _get_sentence(content, m.start(), m.end())
        # Deduplicate — one finding per sentence
        ev_key = evidence[:80].lower()
        if ev_key in seen_sentences:
            continue
        seen_sentences.add(ev_key)
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.PLACEHOLDER,
            detector="placeholder",
            evidence=evidence,
            reason=f'Placeholder found: "{m.group()}"',
        ))
    return findings


def _detect_editorial(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect editorial artifacts — tracked changes, drafts, comments."""
    findings = []

    patterns = [
        # Tracked changes
        (re.compile(r'\[TRACKED CHANGE.*?\]', re.IGNORECASE), "tracked_change",
         "Tracked change marker"),
        # HTML comments
        (re.compile(r'<!--.*?-->', re.DOTALL), "html_comment",
         "HTML comment (hidden content)"),
        # Draft markers
        (re.compile(r'DRAFT\s+v?\d|pending\s+(?:legal\s+)?review|pending\s+approval',
                     re.IGNORECASE), "draft_marker",
         "Draft/review marker"),
        # Revision notes
        (re.compile(r'(?:last edited|previous version|edited by)\s+(?:by\s+)?[A-Z][a-z]+',
                     re.IGNORECASE), "revision_note",
         "Revision note with author"),
        # Strikethrough
        (re.compile(r'~~.+?~~'), "strikethrough",
         "Strikethrough text (deleted content)"),
    ]

    for pattern, detector, description in patterns:
        for m in pattern.finditer(content):
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.EDITORIAL,
                detector=f"editorial_{detector}",
                evidence=m.group()[:100],
                reason=f"{description}: {m.group()[:60]}",
            ))

    return findings


def _detect_metadata_leak(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect internal system metadata exposed in content."""
    findings = []

    patterns = [
        # Jira/project tracking IDs
        (re.compile(r'\b(?:JIRA|TICKET|ISSUE)[-\s]?\d{3,}', re.IGNORECASE),
         "jira_id", "Internal ticket ID"),
        # Internal URLs
        (re.compile(r'https?://(?:[\w-]+\.)?internal\.[\w.-]+[/\w.-]*'),
         "internal_url", "Internal URL"),
        # Admin/debug URLs
        (re.compile(r'https?://(?:admin|debug|staging|dev)\.[\w.-]+[/\w.-]*'),
         "admin_url", "Admin/debug URL"),
        # URLs with debug/verbose mode
        (re.compile(r'https?://[\w.-]+/[\w/]*\?.*(?:debug|verbose|mode=)[\w]*'),
         "debug_url", "URL with debug parameters"),
        # Slack channels/threads
        (re.compile(r'#[a-z][\w-]*(?:-\d{4}-\d{2}(?:-\d{2})?)?', re.IGNORECASE),
         "slack_channel", "Slack channel/thread reference"),
        # Build/version numbers (internal)
        (re.compile(r'(?:build|deploy|release)\s+[\w.-]+-(?:rc|beta|alpha)\d*', re.IGNORECASE),
         "build_number", "Internal build/release number"),
        # Sprint references
        (re.compile(r'Sprint\s+\d+', re.IGNORECASE),
         "sprint_ref", "Sprint reference"),
        # VPN-only resources
        (re.compile(r'VPN\s+required', re.IGNORECASE),
         "vpn_resource", "VPN-restricted resource reference"),
    ]

    for pattern, detector, description in patterns:
        for m in pattern.finditer(content):
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.METADATA_LEAK,
                detector=f"metadata_{detector}",
                evidence=m.group()[:100],
                reason=f"{description}: {m.group()[:60]}",
            ))

    return findings


def _detect_vague(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect vague claims that provide no actionable information."""
    pattern = re.compile(
        r"(?:automatically handles|seamlessly (?:integrates|handles|manages)|"
        r"handles? (?:most|all) (?:edge )?cases|"
        r"takes? care of everything|"
        r"(?:most|all) (?:scenarios|situations|cases) (?:are |will be )?(?:handled|covered|supported)|"
        r"works? (?:out of the box|automatically|without issue)|"
        r"take appropriate action|"
        r"will be (?:handled|addressed|resolved) accordingly)",
        re.IGNORECASE,
    )
    findings = []
    seen_sentences = set()
    for m in pattern.finditer(content):
        evidence = _get_sentence(content, m.start(), m.end())
        ev_key = evidence[:80].lower()
        if ev_key in seen_sentences:
            continue
        seen_sentences.add(ev_key)
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.VAGUE_CLAIM,
            detector="vague_claim",
            evidence=evidence,
            reason=f'Vague claim: "{m.group()}"',
        ))
    return findings


def _detect_destructive(content: str, chunk_id: str,
                        domain_patterns: list = None) -> list[ClassificationFinding]:
    """Detect destructive action instructions."""
    # Base patterns (cross-domain)
    base_pattern = re.compile(
        r"(?:delete|remove permanently|deactivate account|"
        r"drop table|wipe|purge|destroy|erase all|"
        r"reset all|factory reset|format disk)",
        re.IGNORECASE,
    )
    findings = []

    for m in base_pattern.finditer(content):
        evidence = _get_sentence(content, m.start(), m.end())
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.DESTRUCTIVE,
            detector="destructive",
            evidence=evidence,
            reason=f'Destructive action: "{m.group()}"',
        ))

    # Domain-specific patterns from A11
    # Skip patterns that are normal user actions (refund, request)
    _safe_actions = {"request refund", "request a refund", "get a refund"}
    if domain_patterns:
        for dp in domain_patterns:
            if dp.lower() in _safe_actions:
                continue
            # Use word boundary matching, not substring
            dp_re = re.compile(r'\b' + re.escape(dp) + r'\b', re.IGNORECASE)
            m = dp_re.search(content)
            if m:
                evidence = _get_sentence(content, m.start(), m.end())
                findings.append(ClassificationFinding(
                    chunk_id=chunk_id, tag=ChunkTag.DESTRUCTIVE,
                    detector="destructive_domain",
                    evidence=evidence,
                    reason=f'Domain destructive pattern: "{dp}"',
                ))

    return findings


def _detect_broken_ref(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect references to missing or unavailable resources."""
    pattern = re.compile(
        r"(?:see (?:our |the )?(?:FAQ|guide|documentation|handbook|wiki|page)\b|"
        r"refer to (?:the |our )?(?!Figure|Table)\w+|"
        r"image not available|"
        r"link (?:broken|not found|unavailable)|"
        r"\[(?:figure|image|diagram) .+?- (?:image )?not available\])",
        re.IGNORECASE,
    )
    findings = []
    for m in pattern.finditer(content):
        evidence = _get_sentence(content, m.start(), m.end())
        findings.append(ClassificationFinding(
            chunk_id=chunk_id, tag=ChunkTag.BROKEN_REFERENCE,
            detector="broken_reference",
            evidence=evidence,
            reason=f'Broken reference: "{m.group()[:60]}"',
        ))
    return findings


def _detect_escalation(content: str, chunk_id: str) -> list[ClassificationFinding]:
    """Detect escalation instructions with missing details."""
    # First check if escalation is mentioned
    esc_re = re.compile(
        r"escalate to (?:the )?(?:appropriate )?(?:team|management|supervisor)|"
        r"contact (?:them|the team|support)\b(?! (?:at|via|by))",
        re.IGNORECASE,
    )
    findings = []
    for m in esc_re.finditer(content):
        # Check if there's contact details nearby (within 100 chars after)
        after = content[m.end():m.end() + 100]
        has_details = bool(re.search(r'(?:@|phone|email|ext\.|slack|#\w)', after, re.IGNORECASE))
        if not has_details:
            evidence = _get_sentence(content, m.start(), m.end())
            findings.append(ClassificationFinding(
                chunk_id=chunk_id, tag=ChunkTag.ESCALATION,
                detector="escalation_no_details",
                evidence=evidence,
                reason=f'Escalation without contact details: "{m.group()}"',
            ))
    return findings


# =====================================================================
# Helpers
# =====================================================================

def _get_sentence(text: str, start: int, end: int) -> str:
    """Extract the sentence containing the match."""
    # Find sentence start
    s = start
    while s > 0 and text[s - 1] not in '.!?\n':
        s -= 1
    while s < start and text[s] in ' \t\n':
        s += 1

    # Find sentence end
    e = end
    while e < len(text) and text[e] not in '.!?\n':
        e += 1
    if e < len(text) and text[e] in '.!?':
        e += 1

    result = text[s:e].strip()
    return result[:150] if len(result) > 150 else result


# =====================================================================
# Main classifier
# =====================================================================

def _llm_validate_findings(findings: list[ClassificationFinding],
                           content: str, chunk_id: str,
                           llm_call) -> list[ClassificationFinding]:
    """Use LLM to validate governance findings — remove false positives, discover missed issues.

    Sends ALL findings for a chunk in one call. LLM confirms, rejects, or adjusts each.
    """
    import json as _json

    if not findings:
        return findings

    # Build findings list for prompt
    items = []
    for i, f in enumerate(findings):
        items.append(f'{i+1}. [{f.detector}] "{f.evidence[:120]}" — {f.reason[:80]}')
    items_text = "\n".join(items)

    prompt = f"""You are a content governance validator for a customer-facing knowledge base.
Your job is to ensure no unsafe content reaches end users through a RAG retrieval system.

CHUNK CONTENT:
{content[:800]}

RULE-BASED FINDINGS:
{items_text}

VALIDATE each finding. For each one, determine:
- Is this a TRUE governance issue or a FALSE POSITIVE?
- Should the detector type be adjusted?
- How confident are you? (high/medium/low)

GOVERNANCE RULES:
1. PERSONAL PII (block): personal names + their emails/phones. "Sarah Johnson (sarah.johnson@acmecorp.com)" = TRUE PII
2. FUNCTIONAL CONTACTS (allow): support@, info@, billing@ = NOT PII. Published helpdesk numbers = NOT PII
3. FINANCIAL DATA (block): bank account numbers, routing numbers, credit card numbers = PII (type: pii_financial)
4. TICKET/CASE IDs (not PII): "JIRA-7823", "CASE-4521" = metadata_leak, not pii_phone
5. INTERNAL CONTENT (block): "INTERNAL NOTE", "FOR INTERNAL USE ONLY", "do not share with customers"
6. PLACEHOLDERS (block): "TODO", "TBD", "coming soon" = incomplete content
7. EDITORIAL (block): tracked changes, draft markers, strikethrough, revision notes
8. METADATA LEAKS (block): internal URLs, Jira IDs, Slack channels, Sprint references
9. VAGUE CLAIMS (review): "handles most cases", "seamlessly integrates" — only flag if the claim is genuinely unhelpful. Specific descriptions are fine.
10. ESCALATION (review): "escalate to management" without specific contact = TRUE. "contact support team" with details nearby = FALSE POSITIVE

ALSO CHECK: did the rule-based scanner MISS anything in this chunk? If you see PII, internal notes, or unsafe content that wasn't flagged, add it.

Return as JSON array:
[
  {{"finding": 1, "valid": true, "confidence": "high", "reason": "personal email confirmed"}},
  {{"finding": 2, "valid": false, "confidence": "high", "reason": "this is a published support number"}},
  {{"finding": 3, "valid": true, "adjusted_detector": "pii_financial", "confidence": "high", "reason": "bank account number"}},
  {{"finding": "new", "detector": "pii_name", "evidence": "John Smith mentioned in paragraph 2", "reason": "missed personal name"}}
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

        # Apply validations
        validated = []
        mentioned = set()

        for v in validations:
            finding_ref = v.get("finding", 0)

            # New finding discovered by LLM
            if finding_ref == "new":
                detector = v.get("detector", "llm_detected")
                evidence = v.get("evidence", "")
                reason = v.get("reason", "LLM detected")
                # Map detector to tag
                tag_map = {
                    "pii_name": ChunkTag.PII, "pii_email": ChunkTag.PII,
                    "pii_phone": ChunkTag.PII, "pii_financial": ChunkTag.PII,
                    "internal_note": ChunkTag.INTERNAL_ONLY,
                    "metadata_leak": ChunkTag.METADATA_LEAK,
                }
                tag = tag_map.get(detector, ChunkTag.VAGUE_CLAIM)
                validated.append(ClassificationFinding(
                    chunk_id=chunk_id, tag=tag,
                    detector=detector, evidence=evidence, reason=reason,
                ))
                continue

            # Existing finding validation
            idx = finding_ref - 1 if isinstance(finding_ref, int) else -1
            if idx < 0 or idx >= len(findings):
                continue
            mentioned.add(idx)
            f = findings[idx]

            if v.get("valid", True):
                adj = v.get("adjusted_detector", "")
                if adj and adj != f.detector:
                    f.detector = adj
                reason = v.get("reason", "")
                if reason:
                    f.reason = reason
                f.confidence = v.get("confidence", "high")
                validated.append(f)

        # Keep any findings not mentioned (safer to keep than drop)
        for i, f in enumerate(findings):
            if i not in mentioned:
                validated.append(f)

        return validated

    except Exception as e:
        import logging
        logging.getLogger("aiq.a31").warning("LLM validation failed: %s", e)
        return findings


def _remove_flagged_sentences(content: str, findings: list) -> str:
    """Remove sentences containing flagged evidence from content."""
    sentences = re.split(r'(?<=[.!?])\s+', content)
    flagged_evidence = {f.evidence[:50].lower() for f in findings if f.evidence}

    clean = []
    for sent in sentences:
        sent_lower = sent[:50].lower()
        is_flagged = any(ev in sent_lower or sent_lower in ev for ev in flagged_evidence)
        if not is_flagged:
            clean.append(sent)

    return " ".join(clean)


def _llm_remediate_chunk(chunk, findings: list, domain_type: str,
                         llm_call) -> Optional[str]:
    """LLM rewrites a chunk to remove/redact governance issues."""
    issues_list = []
    for f in findings:
        issues_list.append(f'- [{f.tag.value}] "{f.evidence[:120]}" — {f.reason[:60]}')
    issues_text = "\n".join(issues_list)

    prompt = f"""You are a content governance editor for a {domain_type} knowledge base.

This chunk has governance issues that must be fixed before serving to customers.

CHUNK HEADING: {chunk.heading}
CHUNK CONTENT:
{chunk.content}

ISSUES FOUND:
{issues_text}

FIX EACH ISSUE using these rules:
- PII (personal names, emails, phones): Replace with generic description.
  "Sarah Johnson (sarah.johnson@acmecorp.com)" → "A customer"
  "Michael Chen (m.chen@techstart.io, +1-555-0198)" → "A customer"
  Keep the issue description and resolution, remove all personal identity.
- Financial PII (account numbers, routing numbers): Replace with "[redacted]".
  "Account: 7281930456" → "Account: [redacted]"
- Internal content: Remove the internal detail but keep the concept if useful for customers.
  "INTERNAL NOTE: refund threshold $500, agents must get approval" →
  "Refund requests above a certain amount may require additional review."
  If no useful concept remains, remove the sentence entirely.

CRITICAL RULES:
- Do NOT change any sentence that wasn't flagged as an issue
- Keep the chunk structure and flow intact
- The result must read naturally as a knowledge base article
- Every sentence in the output must be safe for customers to read

Return ONLY the cleaned chunk text. Do NOT include "CHUNK HEADING:" or "CHUNK CONTENT:" in your response.
Start directly with the first sentence of the cleaned content."""

    result = llm_call(prompt)
    if result and result.strip():
        cleaned = result.strip()
        # Remove any prompt leakage
        cleaned = re.sub(r'^CHUNK HEADING:.*?\n', '', cleaned)
        cleaned = re.sub(r'^CHUNK CONTENT:\s*\n?', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned.split()) <= len(chunk.content.split()) * 1.3:
            return cleaned
    return None


class Classifier:
    """A31 — Content Governance: decide what is safe to serve."""

    def __init__(self, config: Optional[A31Config] = None):
        self.config = config or A31Config()

    def run(self, chunks: list[Chunk],
            domain_context: Optional[DomainContext] = None) -> ModuleOutput:
        """Classify all chunks by scanning for tag-worthy patterns.

        Args:
            chunks: from A14 (enriched by A21/A22)
            domain_context: from A21 (for destructive patterns)

        Returns:
            ModuleOutput with findings and tagged chunks
        """
        t0 = time.perf_counter()
        words_in = sum(c.words for c in chunks)
        all_findings: list[ClassificationFinding] = []
        tagged_count = 0

        domain_destructive = []
        if domain_context:
            domain_destructive = domain_context.destructive_patterns

        for chunk in chunks:
            content = chunk.content
            chunk_findings: list[ClassificationFinding] = []

            # Run all enabled detectors
            if self.config.detect_internal:
                chunk_findings.extend(_detect_internal(content, chunk.chunk_id))
            if self.config.detect_pii:
                chunk_findings.extend(_detect_pii(
                    content, chunk.chunk_id,
                    pii_mode=self.config.pii_mode,
                    safe_prefixes=self.config.safe_email_prefixes,
                ))
            if self.config.detect_placeholder:
                chunk_findings.extend(_detect_placeholder(content, chunk.chunk_id))
            if self.config.detect_editorial:
                chunk_findings.extend(_detect_editorial(content, chunk.chunk_id))
            if self.config.detect_metadata_leak:
                chunk_findings.extend(_detect_metadata_leak(content, chunk.chunk_id))
            if self.config.detect_vague:
                chunk_findings.extend(_detect_vague(content, chunk.chunk_id))
            if self.config.detect_destructive:
                chunk_findings.extend(_detect_destructive(content, chunk.chunk_id, domain_destructive))
            if self.config.detect_broken_ref:
                chunk_findings.extend(_detect_broken_ref(content, chunk.chunk_id))
            if self.config.detect_escalation:
                chunk_findings.extend(_detect_escalation(content, chunk.chunk_id))

            # LLM validation — validate ambiguous findings with context
            if self.config.llm_call and chunk_findings:
                chunk_findings = _llm_validate_findings(
                    chunk_findings, content, chunk.chunk_id, self.config.llm_call)

            all_findings.extend(chunk_findings)

            # Assign tag: most severe finding wins
            if chunk_findings:
                best = min(chunk_findings, key=lambda f: _TAG_PRIORITY.get(f.tag, 99))
                chunk.tag = best.tag
                chunk.tag_reason = best.reason
                chunk.tag_module = "A31"
                tagged_count += 1

                # Track as "removed" tokens (content classified, not served to users)
                if best.tag.default_behavior == "block":
                    chunk.token_changes.append(TokenChange(
                        change_type="removed",
                        reason=f"tagged_{best.tag.value}",
                        token_count=chunk.words,
                        module="A31",
                        detail=best.reason[:60],
                    ))

        detected = len(all_findings)

        return ModuleOutput(
            module_id="A31",
            module_name="Content Governance",
            detected=detected,
            resolved=tagged_count,
            remaining=0,
            words_in=words_in,
            words_out=words_in,  # content unchanged, only tags added
            findings=all_findings,
            chunks=chunks,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def remediate(self, chunks: list[Chunk], findings: list,
                  domain_context: Optional[DomainContext] = None) -> dict:
        """Fix governance issues in chunks — remove noise, rewrite sensitive content.

        For each tagged chunk:
          - Noise tags (placeholder, editorial, metadata_leak): remove flagged sentences
          - Content tags (internal_only, pii): LLM rewrites to redact sensitive parts
          - Review tags (vague_claim, broken_reference, escalation): leave as-is

        Returns dict with stats: {fixed, removed, unchanged, errors}
        """
        stats = {"fixed": 0, "removed": 0, "unchanged": 0, "errors": 0}
        domain_type = domain_context.domain_type if domain_context else "general"

        for chunk in chunks:
            if chunk.tag.value == "content":
                continue

            chunk_findings = [f for f in findings if f.chunk_id == chunk.chunk_id]
            if not chunk_findings:
                continue

            # Review tags — don't modify, user decides
            if chunk.tag.default_behavior == "review":
                stats["unchanged"] += 1
                continue

            # Noise tags — remove sentences (no LLM needed)
            noise_tags = {"placeholder", "editorial", "metadata_leak"}
            if chunk.tag.value in noise_tags:
                cleaned = _remove_flagged_sentences(chunk.content, chunk_findings)
                if cleaned and cleaned.strip() and len(cleaned.split()) >= 10:
                    chunk.content = cleaned.strip()
                    chunk.words = len(chunk.content.split())
                    chunk.tag = ChunkTag.CONTENT
                    chunk.tag_reason = f"Cleaned: removed {chunk.tag.value} content"
                    chunk.tag_module = "A31 (remediated)"
                    stats["removed"] += 1
                else:
                    stats["unchanged"] += 1
                continue

            # Content tags (pii, internal_only) — LLM rewrite
            if self.config.llm_call and chunk.tag.value in ("pii", "internal_only"):
                try:
                    rewritten = _llm_remediate_chunk(
                        chunk, chunk_findings, domain_type, self.config.llm_call)
                    if rewritten:
                        chunk.content = rewritten
                        chunk.words = len(rewritten.split())
                        chunk.tag = ChunkTag.CONTENT
                        chunk.tag_reason = f"Remediated: {chunk.tag.value} content rewritten"
                        chunk.tag_module = "A31 (remediated)"
                        stats["fixed"] += 1
                    else:
                        stats["unchanged"] += 1
                except Exception:
                    stats["errors"] += 1
                continue

            stats["unchanged"] += 1

        return stats
