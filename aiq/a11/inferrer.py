"""A11 — Domain Intelligence

Builds the shared context layer consumed by ALL downstream modules.
Scans content to infer domain type, extract vocabulary, map entities,
and identify scope qualifiers.

What it extracts:
    - Domain type (support, medical, legal, engineering, finance, hr)
    - Company/organization name
    - Actor/team vocabulary (who handles what)
    - Acronym dictionary (with expansions)
    - Product/service catalog
    - Scope qualifiers (region, tier, version)
    - Destructive action patterns (delete account, cancel subscription)
    - Entity relationships (resolves "they" -> specific team)

How it works:
    Three modes with progressive quality/cost:
    - rule_only:     Regex + frequency analysis. Free, no API calls.
    - rule_then_llm: Rule-based first, one LLM call to validate/enrich.
    - llm_all:       LLM extracts from scratch, rule-based fills gaps.

    All modes produce the same DomainContext object. LLM modes fall back
    to rule-based if the LLM call fails.

Config:
    mode: "rule_only" | "rule_then_llm" | "llm_all" (default: "rule_only")
    llm_call: optional callable(prompt: str) -> str
    source_title: document title for context (default: "")
    min_topic_frequency: minimum word frequency to become a topic anchor (default: 2)
    min_actor_frequency: minimum mention frequency to become an actor (default: 2)

Config exposed to AIQConfig:
    domain_mode           -> A11Config.mode                (default: "rule_only")
    domain_source_title   -> A11Config.source_title        (default: "")
    (llm_call is wired internally from AIQConfig.llm_client)

User overrides (applied after auto-detection, via AIQConfig):
    domain_type       -> replaces auto-detected domain type (e.g. "medical")
    company_name      -> replaces auto-detected company name
    extra_actors      -> dict merged into detected actors (e.g. {"radiology": "radiology"})
    extra_acronyms    -> dict merged into detected acronyms (e.g. {"MRN": "Medical Record Number"})
    extra_products    -> list merged into detected products (e.g. ["Premium Plan"])
    extra_destructive -> list merged into detected destructive patterns

    Override = replaces auto-detected value entirely.
    Extra    = merged on top of auto-detected values (additive).
    If no overrides/extras provided, auto-detection is used as-is.

Input:  list[Chunk]
Output: ModuleOutput with .data["domain_context"] = DomainContext

LLM required: No (rule_only mode). Optional for enrichment.
"""
from __future__ import annotations

import re
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import Chunk, DomainContext, ModuleOutput


# =====================================================================
# Config
# =====================================================================

@dataclass
class A11Config:
    """Configuration for domain context extraction."""
    mode: str = "rule_only"  # "rule_only" | "rule_then_llm" | "llm_all"
    llm_call: Optional[Callable] = None
    source_title: str = ""
    min_topic_frequency: int = 2
    min_actor_frequency: int = 2


# =====================================================================
# Domain signals & defaults
# =====================================================================

_DOMAIN_SIGNALS = {
    "support": {
        "support", "ticket", "agent", "escalation", "escalate", "customer",
        "resolve", "resolution", "case", "refund", "billing", "subscription",
        "troubleshoot", "issue", "help", "assist",
    },
    "medical": {
        "patient", "diagnosis", "treatment", "medication", "dosage", "symptom",
        "clinical", "prescription", "therapy", "hospital", "doctor", "nurse",
    },
    "legal": {
        "contract", "clause", "liability", "compliance", "regulation", "statute",
        "agreement", "obligation", "breach", "indemnity", "arbitration",
    },
    "engineering": {
        "api", "endpoint", "deployment", "server", "database", "schema",
        "migration", "microservice", "repository", "pipeline",
    },
    "finance": {
        "revenue", "budget", "forecast", "investment", "portfolio", "asset",
        "liability", "equity", "quarterly", "fiscal", "dividend",
    },
    "hr": {
        "employee", "onboarding", "payroll", "benefits", "performance",
        "recruitment", "compensation", "termination", "leave",
    },
}

DOMAIN_DEFAULTS = {
    "support": {
        "acronyms": {
            "CRM": "Customer Relationship Management",
            "SLA": "Service Level Agreement",
            "CSM": "Customer Success Manager",
            "NPS": "Net Promoter Score",
            "CSAT": "Customer Satisfaction Score",
        },
        "actors": {"support": "support", "billing": "billing", "finance": "finance"},
    },
    "medical": {
        "acronyms": {"HIPAA": "Health Insurance Portability and Accountability Act",
                     "PHI": "Protected Health Information", "EHR": "Electronic Health Record"},
        "actors": {"doctor": "physician", "nurse": "nursing", "patient": "patient"},
    },
    "legal": {
        "acronyms": {"NDA": "Non-Disclosure Agreement", "GDPR": "General Data Protection Regulation"},
        "actors": {"counsel": "legal", "compliance": "compliance"},
    },
    "engineering": {
        "acronyms": {"API": "Application Programming Interface", "SDK": "Software Development Kit",
                     "CI": "Continuous Integration"},
        "actors": {"devops": "devops", "engineering": "engineering"},
    },
    "finance": {
        "acronyms": {"ROI": "Return on Investment", "KPI": "Key Performance Indicator"},
        "actors": {"cfo": "finance", "controller": "finance"},
    },
    "hr": {
        "acronyms": {"PTO": "Paid Time Off", "HRIS": "Human Resources Information System"},
        "actors": {"recruiter": "hr", "hr": "hr"},
    },
    "general": {"acronyms": {}, "actors": {}},
}


def get_domain_defaults(domain_type: str) -> dict:
    return DOMAIN_DEFAULTS.get(domain_type, DOMAIN_DEFAULTS["general"])


# =====================================================================
# Shared constants
# =====================================================================

_STOP_WORDS = frozenset(
    "the and for with from this that may also not but are was were "
    "has have had been being will can could should would all some any "
    "its into our your their you due must need use used make makes "
    "get gets set sets take takes give gives work works keep keeps "
    "see check try follow contact please note ensure provide include "
    "new old more less than each per every about between after before "
    "then when where which what how who why other another such same "
    "just only still even much many most very well too quite always "
    "never usually often sometimes already yet again both either "
    "these those here there above below next last first second "
    "information details section page steps process issues "
    "using available based currently following within without".split()
)

_FALSE_ACRONYMS = {
    "FOR", "AND", "THE", "NOT", "ALL", "USE", "SET", "GET", "PUT",
    "ADD", "RUN", "LET", "TRY", "ANY", "OUT", "OLD", "NEW", "NOW",
    "CASE", "NOTE", "ONLY", "ALSO", "MUST", "WILL", "WHEN", "WITH",
    "FROM", "THAT", "THIS", "THEY", "THEM", "HAVE", "BEEN", "EACH",
    "DOES", "DONE", "MAKE", "TAKE", "GIVE", "KEEP", "WORK", "FIND",
    "DRAFT", "CHANGE", "LEVEL", "STEP", "CALL", "PLAN", "CLICK",
    "MARCH", "APRIL", "JUNE", "JULY",
}

_KNOWN_EXPANSIONS = {
    "CRM": "Customer Relationship Management",
    "SLA": "Service Level Agreement",
    "SSO": "Single Sign-On",
    "PO": "Purchase Order",
    "DPO": "Data Protection Officer",
    "VPN": "Virtual Private Network",
    "CSM": "Customer Success Manager",
    "SWIFT": "Society for Worldwide Interbank Financial Telecommunication",
    "CVV": "Card Verification Value",
    "PII": "Personally Identifiable Information",
    "MFA": "Multi-Factor Authentication",
    "KPI": "Key Performance Indicator",
}

_COMMON_TITLE_WORDS = {
    "The", "This", "That", "These", "Those", "When", "Where",
    "What", "How", "For", "All", "Our", "Your", "Step", "Figure",
    "Table", "Section", "Contact", "Method", "Status", "Level",
    "Feature", "Code", "Field", "Value", "Issue", "Resolution",
    "Case", "Type", "Note", "Process", "Summary", "Content",
    "Custom", "Active", "None", "Yes", "March", "April", "May",
    "June", "July", "August", "January", "February", "September",
    "October", "November", "December",
}


# =====================================================================
# Content sampling — representative preview across entire document
# =====================================================================

def _sample_content(chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Sample content from across the entire document.

    Strategy: include ALL chunks but truncate each proportionally
    so the LLM sees every topic. Better than skipping chunks entirely.
    """
    if not chunks:
        return ""

    total_content = sum(len(c.content) for c in chunks)
    if total_content <= max_chars:
        return "\n\n---\n\n".join(c.content for c in chunks)

    # Proportional truncation: each chunk gets budget relative to its size
    # but with a minimum of 200 chars per chunk
    min_per_chunk = 200
    reserved = min_per_chunk * len(chunks)
    remaining_budget = max(0, max_chars - reserved)

    parts = []
    for chunk in chunks:
        proportion = len(chunk.content) / total_content if total_content else 0
        budget = min_per_chunk + int(remaining_budget * proportion)
        parts.append(chunk.content[:budget])

    return "\n\n---\n\n".join(parts)


# =====================================================================
# Rule-based extractors (Local mode)
# =====================================================================

def _extract_word_freq(chunks: list[Chunk]) -> Counter:
    counter = Counter()
    for chunk in chunks:
        for w in re.findall(r'\b[a-z]{4,}\b', chunk.content.lower()):
            if w not in _STOP_WORDS:
                counter[w] += 1
    return counter


def _infer_domain_type(word_freq: Counter) -> tuple[str, float]:
    scores = {}
    for domain, signals in _DOMAIN_SIGNALS.items():
        scores[domain] = sum(word_freq.get(w, 0) for w in signals)
    total = sum(scores.values())
    if total == 0:
        return "general", 0.0
    best = max(scores, key=scores.get)
    return best, round(scores[best] / total, 2)


def _extract_company_name(source_title: str, chunks: list[Chunk]) -> str:
    if source_title:
        parts = re.split(r'\s*[-–—|:]\s*', source_title)
        if parts:
            candidate = parts[0].strip()
            if candidate and candidate[0].isupper() and candidate.lower() not in _STOP_WORDS:
                return candidate
    name_counter = Counter()
    for chunk in chunks:
        for name in re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b', chunk.content):
            if name not in _COMMON_TITLE_WORDS and len(name) > 3:
                name_counter[name] += 1
    if name_counter:
        best, count = name_counter.most_common(1)[0]
        if count >= 3:
            return best
    return ""


def _extract_actors(chunks: list[Chunk], min_freq: int) -> dict:
    actor_counter = Counter()
    # Pattern 1: "X team/department"
    team_re = re.compile(
        r'\b(?:the\s+)?(\w+)\s+(?:team|department|group|division)\b', re.IGNORECASE)
    # Pattern 2: "X handles/manages/processes"
    handler_re = re.compile(
        r'\b(\w+)\s+(?:handles?|manages?|processes?|reviews?|approves?)\b', re.IGNORECASE)
    # Pattern 3: "escalate to / contact X"
    escalate_re = re.compile(
        r'\b(?:escalate\s+to|contact)\s+(?:the\s+)?(\w+)\s*(?:team|department)?\b', re.IGNORECASE)
    # Pattern 4: "X inquiries" or "X escalations" (e.g., "billing inquiries")
    topic_re = re.compile(
        r'\b(\w+)\s+(?:inquiries|escalations|disputes|issues)\b', re.IGNORECASE)

    false_actors = {
        "the", "this", "that", "them", "they", "your", "our",
        "account", "system", "platform", "it", "out", "issue",
        "management", "appropriate", "relevant", "customer",
        "payment", "refund", "all", "any", "most",
    }
    for chunk in chunks:
        for pattern in [team_re, handler_re, escalate_re, topic_re]:
            for m in pattern.finditer(chunk.content):
                actor = m.group(1).strip().lower()
                if actor and len(actor) > 2 and actor not in false_actors:
                    actor_counter[actor] += 1
    return {a: a for a, c in actor_counter.items() if c >= min_freq}


def _extract_acronyms(chunks: list[Chunk], common: set) -> dict:
    acronym_re = re.compile(r'\b([A-Z]{2,5})\b')
    acronym_counter = Counter()
    for chunk in chunks:
        for m in acronym_re.finditer(chunk.content):
            acr = m.group(1)
            if acr not in common and acr not in _FALSE_ACRONYMS:
                after = chunk.content[m.start():m.start() + len(acr) + 5]
                if re.match(r'[A-Z]+-\d', after):
                    continue
                acronym_counter[acr] += 1

    expansion_re = re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,6})\)'
        r'|([A-Z]{2,6})\s*\(([A-Z][a-z]+(?:\s+[a-zA-Z]+)*)\)',
    )
    expansions = {}
    for chunk in chunks:
        for m in expansion_re.finditer(chunk.content):
            if m.group(1) and m.group(2):
                expansions[m.group(2)] = m.group(1)
            elif m.group(3) and m.group(4):
                expansions[m.group(3)] = m.group(4)

    result = {}
    for acr, count in acronym_counter.items():
        if acr in expansions:
            result[acr] = expansions[acr]
        elif acr in _KNOWN_EXPANSIONS:
            result[acr] = _KNOWN_EXPANSIONS[acr]
        elif count >= 3:
            result[acr] = "unknown"
    return result


def _extract_products_strict(chunks: list[Chunk], company_name: str) -> list[str]:
    # Pattern 1: "X plan/tier/subscription"
    plan_re = re.compile(
        r'\b(Starter|Professional|Enterprise|Basic|Premium|Standard|Pro|Free)\s+'
        r'(?:plan|tier|edition|package|subscription|account|trial)\b', re.IGNORECASE)
    products = set()
    for chunk in chunks:
        for m in plan_re.finditer(chunk.content):
            products.add(m.group(1).title())

    # Pattern 2: capitalized word near "$X/month"
    pricing_re = re.compile(
        r'\b([A-Z][a-z]+)\b[^.]{0,30}\$\d+(?:\.\d+)?(?:/(?:mo|month|year|yr))')
    for chunk in chunks:
        for m in pricing_re.finditer(chunk.content):
            candidate = m.group(1)
            if candidate not in _COMMON_TITLE_WORDS and candidate != company_name:
                products.add(candidate)

    # Pattern 3: "X customers" or "X pricing" where X is a known tier
    tier_context_re = re.compile(
        r'\b(Starter|Professional|Enterprise)\s+'
        r'(?:customers?|users?|pricing|features?)\b', re.IGNORECASE)
    for chunk in chunks:
        for m in tier_context_re.finditer(chunk.content):
            products.add(m.group(1).title())

    # Remove "Free" — it's usually "free trial", not a paid product
    products.discard("Free")

    return sorted(products)


def _extract_scope_qualifiers(chunks: list[Chunk]) -> list[dict]:
    qualifiers = []
    regions = set()
    for chunk in chunks:
        for m in re.finditer(r'\b(North America|Europe|Asia|US|UK|EU|APAC|EMEA)\b',
                             chunk.content, re.IGNORECASE):
            regions.add(m.group().title())
    if regions:
        qualifiers.append({"type": "region", "values": sorted(regions)})
    tiers = set()
    for chunk in chunks:
        for m in re.finditer(
                r'\b(Starter|Professional|Enterprise|Basic|Premium|Standard|Free|Pro)\b',
                chunk.content, re.IGNORECASE):
            tiers.add(m.group().title())
    if tiers:
        qualifiers.append({"type": "tier", "values": sorted(tiers)})
    versions = set()
    for chunk in chunks:
        for m in re.finditer(r'\b[Vv](?:ersion)?\s*(\d+(?:\.\d+)*)\b', chunk.content):
            versions.add(f"v{m.group(1)}")
    if versions:
        qualifiers.append({"type": "version", "values": sorted(versions)})
    return qualifiers


def _extract_destructive_patterns(chunks: list[Chunk]) -> list[str]:
    patterns = []
    seen = set()

    # Pattern 1: "verb + object" (delete account, cancel subscription)
    direct_re = re.compile(
        r'\b(delete|cancel|deactivate|revoke|disable|remove|terminate|close)\s+'
        r'(?:your\s+|the\s+|my\s+|an?\s+)?'
        r'(account|subscription|membership|access|service|plan|profile|data)\b',
        re.IGNORECASE)

    # Pattern 2: "can cancel/delete at any time" → extract the verb + context
    context_re = re.compile(
        r'\b(?:can|may|to)\s+(cancel|delete|deactivate|terminate|close)\b'
        r'[^.]{0,30}\b(subscription|account|plan|service|membership)\b',
        re.IGNORECASE)

    # Pattern 3: "request deletion/cancellation" (NOT refund — refunds are normal user actions)
    request_re = re.compile(
        r'\brequest\s+(?:a\s+)?(deletion|cancellation|termination)\b',
        re.IGNORECASE)

    for chunk in chunks:
        for m in direct_re.finditer(chunk.content):
            phrase = f"{m.group(1).lower()} {m.group(2).lower()}"
            if phrase not in seen:
                seen.add(phrase)
                patterns.append(phrase)
        for m in context_re.finditer(chunk.content):
            phrase = f"{m.group(1).lower()} {m.group(2).lower()}"
            if phrase not in seen:
                seen.add(phrase)
                patterns.append(phrase)
        for m in request_re.finditer(chunk.content):
            phrase = f"request {m.group(1).lower()}"
            if phrase not in seen:
                seen.add(phrase)
                patterns.append(phrase)

    return patterns


def _extract_entity_relationships(chunks: list[Chunk], actors: dict,
                                  company_name: str) -> dict:
    """Map vague references to specific entities per context section.

    Scans each chunk for pronouns ("they", "them") and vague references
    ("the team", "the platform") and maps them to the most likely entity
    based on surrounding content.
    """
    relationships = {}
    actor_set = set(actors.keys())

    for chunk in chunks:
        content_lower = chunk.content.lower()
        # Determine context topic from chunk content
        context_key = ""
        for actor in actor_set:
            if actor in content_lower:
                context_key = f"{actor} context"
                break
        if not context_key:
            # Try heading
            heading_words = set(re.findall(r'\b[a-z]{4,}\b',
                                           (chunk.heading or "").lower()))
            for actor in actor_set:
                if actor in heading_words:
                    context_key = f"{actor} context"
                    break
        if not context_key:
            continue

        mapping = relationships.setdefault(context_key, {})

        # Check for pronoun usage
        if re.search(r'\bthey\b|\bthem\b|\btheir\b', content_lower):
            # Find the most mentioned actor in this chunk
            actor_counts = {}
            for a in actor_set:
                actor_counts[a] = len(re.findall(rf'\b{a}\b', content_lower))
            best_actor = max(actor_counts, key=actor_counts.get, default="")
            if best_actor and actor_counts[best_actor] > 0:
                mapping["they"] = f"{best_actor} team"
                mapping["them"] = f"{best_actor} team"

        # Vague entity references
        if "the platform" in content_lower or "the system" in content_lower:
            if company_name:
                mapping["the platform"] = company_name
                mapping["the system"] = company_name

        if "the team" in content_lower:
            for a in actor_set:
                if a in content_lower:
                    mapping["the team"] = f"{a} team"
                    break

    return relationships


def _build_topic_anchors(word_freq: Counter, min_freq: int) -> dict:
    frequent = {w for w, c in word_freq.items() if c >= min_freq}
    groups: dict[str, set] = {}
    for word in sorted(frequent):
        matched = False
        for root, members in groups.items():
            min_prefix = min(len(word), len(root), 6)
            if min_prefix >= 6 and word[:min_prefix] == root[:min_prefix]:
                members.add(word)
                matched = True
                break
            if (len(word) >= 5 and len(root) >= 5
                    and (word.startswith(root) or root.startswith(word))):
                members.add(word)
                matched = True
                break
        if not matched:
            groups[word] = {word}
    anchors = {}
    for root, members in groups.items():
        best = max(members, key=lambda w: word_freq[w])
        anchors[best] = members
    return anchors


# =====================================================================
# LLM prompts
# =====================================================================

def _build_llm_extract_prompt(content_preview: str, source_title: str) -> str:
    return f"""You are building a domain context layer for a knowledge base quality system.
This context will be used to:
- Resolve ambiguous pronouns ("they" → which team?)
- Expand acronyms readers might not know
- Detect when content applies to different regions/tiers/audiences
- Identify dangerous actions that need safety warnings

DOCUMENT TITLE: {source_title}

DOCUMENT CONTENT (sampled across all sections):
{content_preview}

Extract the following as JSON:

{{
  "domain_type": "support|medical|legal|engineering|finance|hr|general — pick based on the PRIMARY purpose of this document",
  "company_name": "the organization that owns this KB — look in title, headers, brand mentions",
  "products": [
    {{"name": "plan/product name", "context": "brief description — e.g. '$29/mo, up to 5 users'"}}
  ],
  "teams": [
    {{"name": "team name", "role": "what they do — e.g. 'handles refund escalations'"}}
  ],
  "acronyms": [
    {{"acronym": "ABC", "expansion": "Full Name", "context": "where/how it's used in this doc"}}
  ],
  "scope_qualifiers": [
    {{"type": "region|tier|version|audience", "values": ["value1", "value2"], "reason": "why these matter — e.g. 'different pricing per region'"}}
  ],
  "destructive_patterns": ["verb + object — only user-triggered irreversible actions"],
  "entity_relationships": [
    {{"vague_term": "they/the team/the platform", "resolves_to": "specific entity", "context": "in what section/topic"}}
  ]
}}

RULES:
- domain_type: "support" = helps customers with billing/payments/troubleshooting/refunds/subscriptions. "finance" = investment portfolio management/budgeting/forecasting. A payment & billing KB is SUPPORT, not finance.
- company_name: the organization, NOT a product tier name.
- products: only things a customer can BUY or SUBSCRIBE to. Include ALL tiers/plans mentioned (e.g., Starter, Professional, Enterprise). Include pricing if mentioned. "March" = no. "Billing" = no (that's a department).
- teams: real organizational units mentioned as handling work. "support team" = yes. "them" = no. Include what each team DOES.
- acronyms: uppercase 2-6 letter terms that a reader might not know. Skip: error codes (ERR-001), Jira keys (PAY-1234), common words (FOR, AND), currencies (USD, EUR). Include WHERE it appears in the doc.
- scope_qualifiers: only if the document has DIFFERENT rules/pricing/policies per segment. Include WHY it matters.
- destructive_patterns: user-triggered irreversible actions. "cancel subscription" = yes. "system processes refund" = no.
- entity_relationships: map EVERY "they", "them", "the team", "the platform", "the system" to what it actually refers to in that section.

Return ONLY valid JSON."""


def _build_llm_validate_prompt(candidates: dict, content_preview: str) -> str:
    candidates_text = json.dumps(candidates, indent=2, default=str)
    return f"""You are validating and enriching extracted metadata from a knowledge base.

DOCUMENT CONTENT (sampled):
{content_preview[:2000]}

RULE-BASED CANDIDATES:
{candidates_text}

TASKS:
1. VALIDATE: Remove noise (false products, wrong acronyms, bad actors)
2. ENRICH: Add anything the rules missed — especially teams, products, and entity_relationships
3. FIX: Correct any wrong expansions or classifications
4. DO NOT change domain_type — the rule-based detection is reliable for this

Return clean results as JSON:

{{
  "domain_type": "support|medical|legal|engineering|finance|hr|general",
  "company_name": "the organization name",
  "products": [{{"name": "...", "context": "..."}}],
  "teams": [{{"name": "...", "role": "what they do"}}],
  "acronyms": [{{"acronym": "...", "expansion": "...", "context": "..."}}],
  "scope_qualifiers": [{{"type": "...", "values": [...], "reason": "..."}}],
  "destructive_patterns": ["..."],
  "entity_relationships": [{{"vague_term": "...", "resolves_to": "...", "context": "..."}}]
}}

VALIDATION RULES:
- Products: REMOVE months (March), common nouns (Click, Check), adjectives. KEEP named plans with pricing.
- Teams: REMOVE pronouns, generic phrases. KEEP organizational units. ADD their role/responsibility.
- Acronyms: REMOVE error codes, Jira keys, common words. KEEP domain-specific terms. ADD missing expansions.
- Entity relationships: For EVERY "they/them/the team/the platform/the system" in the content, map it to the specific entity in that context. This is critical for downstream pronoun resolution.

Return ONLY valid JSON."""


# =====================================================================
# LLM result parsing
# =====================================================================

def _parse_llm_result(result_text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not result_text:
        return {}
    # Strip markdown code blocks
    text = result_text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, Exception):
        pass
    return {}


def _merge_acronyms(rule_based: dict, llm_based: dict) -> dict:
    """Merge acronyms: keep all rule-based, add LLM expansions for unknowns."""
    merged = dict(rule_based)
    for acr, expansion in llm_based.items():
        if acr not in merged:
            merged[acr] = expansion
        elif merged[acr] == "unknown" and expansion and expansion != "unknown":
            merged[acr] = expansion
    return merged


def _llm_expand_acronyms(unknowns: dict, content_preview: str,
                         llm_call: Callable) -> dict:
    """Use LLM to expand unknown acronyms found by rule-based extraction."""
    acr_list = ", ".join(unknowns.keys())
    prompt = f"""These acronyms appear in a knowledge base document but their expansions are unknown.
Based on the document context, provide the full expansion for each.

ACRONYMS: {acr_list}

DOCUMENT CONTEXT:
{content_preview[:1500]}

Return as JSON: {{"ACRONYM": "Full Expansion"}}
If you cannot determine the expansion, use "unknown".
Return ONLY the JSON."""

    try:
        result = llm_call(prompt)
        parsed = _parse_llm_result(result)
        if parsed:
            return {k: v for k, v in parsed.items() if v and v != "unknown"}
    except Exception:
        pass
    return {}


def _llm_to_context(extracted: dict, anchors: dict, normalization: dict,
                    fallback_company: str) -> DomainContext:
    """Convert LLM JSON output to DomainContext."""
    # Parse products
    products = []
    for p in extracted.get("products", []):
        if isinstance(p, dict):
            products.append(p.get("name", ""))
        elif isinstance(p, str):
            products.append(p)
    products = [p for p in products if p]

    # Parse teams
    actors = {}
    for t in extracted.get("teams", []):
        if isinstance(t, dict):
            name = t.get("name", "").lower()
            if name:
                actors[name] = name
        elif isinstance(t, str):
            actors[t.lower()] = t.lower()

    # Parse acronyms
    acronyms = {}
    for a in extracted.get("acronyms", []):
        if isinstance(a, dict):
            acr = a.get("acronym", "")
            exp = a.get("expansion", "unknown")
            if acr:
                acronyms[acr] = exp
        elif isinstance(a, str):
            acronyms[a] = "unknown"

    return DomainContext(
        domain_type=extracted.get("domain_type", "general"),
        confidence=0.9,
        company_name=extracted.get("company_name", fallback_company),
        domain_anchors=anchors,
        actors=actors,
        acronyms=acronyms,
        product_names=products,
        context_normalization=normalization,
        destructive_patterns=extracted.get("destructive_patterns", []),
        scope_qualifiers=extracted.get("scope_qualifiers", []),
    )


# =====================================================================
# Main inferrer
# =====================================================================

class DomainInferrer:
    """A11 — Domain Context: build the context layer for the pipeline."""

    def __init__(self, config: Optional[A11Config] = None):
        self.config = config or A11Config()

    def run(self, chunks: list[Chunk]) -> ModuleOutput:
        """Scan all chunks and produce a DomainContext."""
        t0 = time.perf_counter()
        words_in = sum(c.words for c in chunks)
        content_preview = _sample_content(chunks)

        if self.config.mode == "llm_all" and self.config.llm_call:
            ctx = self._run_api_mode(chunks, content_preview)
        elif self.config.mode == "rule_then_llm" and self.config.llm_call:
            ctx = self._run_hybrid_mode(chunks, content_preview)
        else:
            ctx = self._run_local_mode(chunks)

        detected = (
            len(ctx.domain_anchors) + len(ctx.actors) + len(ctx.acronyms)
            + len(ctx.product_names) + len(ctx.scope_qualifiers)
            + len(ctx.destructive_patterns)
        )

        return ModuleOutput(
            module_id="A11",
            module_name="Domain Context",
            detected=detected,
            resolved=detected,
            remaining=0,
            words_in=words_in,
            words_out=words_in,
            elapsed_seconds=time.perf_counter() - t0,
            data={"domain_context": ctx},
        )

    def _run_local_mode(self, chunks: list[Chunk]) -> DomainContext:
        word_freq = _extract_word_freq(chunks)
        domain_type, confidence = _infer_domain_type(word_freq)
        company_name = _extract_company_name(self.config.source_title, chunks)
        anchors = _build_topic_anchors(word_freq, self.config.min_topic_frequency)
        actors = _extract_actors(chunks, self.config.min_actor_frequency)
        common = DomainContext().common_acronyms
        acronyms = _extract_acronyms(chunks, common)
        products = _extract_products_strict(chunks, company_name)
        scope = _extract_scope_qualifiers(chunks)
        destructive = _extract_destructive_patterns(chunks)
        normalization = {v: k for k, vs in anchors.items() for v in vs if v != k}
        return DomainContext(
            domain_type=domain_type,
            confidence=confidence,
            company_name=company_name,
            domain_anchors=anchors,
            actors=actors,
            acronyms=acronyms,
            product_names=products,
            context_normalization=normalization,
            destructive_patterns=destructive,
        scope_qualifiers=scope,
    )

    def _run_hybrid_mode(self, chunks: list[Chunk],
                         content_preview: str) -> DomainContext:
        # Rule-based first
        ctx = self._run_local_mode(chunks)

        # Build candidates for LLM validation
        candidates = {
            "domain_type": ctx.domain_type,
            "company_name": ctx.company_name,
            "products": [{"name": p} for p in ctx.product_names],
            "teams": [{"name": a, "role": ""} for a in ctx.actors],
            "acronyms": [{"acronym": k, "expansion": v} for k, v in ctx.acronyms.items()],
            "scope_qualifiers": ctx.scope_qualifiers,
            "destructive_patterns": ctx.destructive_patterns,
        }

        prompt = _build_llm_validate_prompt(candidates, content_preview)
        try:
            result_text = self.config.llm_call(prompt)
            cleaned = _parse_llm_result(result_text)
        except Exception:
            cleaned = {}

        if cleaned:
            enriched = _llm_to_context(
                cleaned, ctx.domain_anchors, ctx.context_normalization,
                ctx.company_name,
            )
            enriched.domain_anchors = ctx.domain_anchors
            enriched.context_normalization = ctx.context_normalization
            if ctx.confidence >= 0.7:
                enriched.domain_type = ctx.domain_type
            enriched.confidence = ctx.confidence
            if not enriched.company_name:
                enriched.company_name = ctx.company_name

            # ALWAYS keep rule-based acronyms (LLM is bad at finding them)
            # But use LLM expansions for unknowns
            enriched.acronyms = _merge_acronyms(ctx.acronyms, enriched.acronyms)

            return enriched

        return ctx

    def _run_api_mode(self, chunks: list[Chunk],
                      content_preview: str) -> DomainContext:
        # Rule-based for anchors AND acronyms (LLM is bad at both)
        word_freq = _extract_word_freq(chunks)
        anchors = _build_topic_anchors(word_freq, self.config.min_topic_frequency)
        normalization = {v: k for k, vs in anchors.items() for v in vs if v != k}
        fallback_company = _extract_company_name(self.config.source_title, chunks)
        common = DomainContext().common_acronyms
        rule_acronyms = _extract_acronyms(chunks, common)

        # LLM extracts everything else
        prompt = _build_llm_extract_prompt(content_preview, self.config.source_title)
        try:
            result_text = self.config.llm_call(prompt)
            extracted = _parse_llm_result(result_text)
        except Exception:
            extracted = {}

        if extracted:
            ctx = _llm_to_context(extracted, anchors, normalization, fallback_company)
            # Always use rule-based acronyms, merge with any LLM found
            ctx.acronyms = _merge_acronyms(rule_acronyms, ctx.acronyms)
            # Use LLM to expand unknowns
            unknowns = {k: v for k, v in ctx.acronyms.items() if v == "unknown"}
            if unknowns and self.config.llm_call:
                expanded = _llm_expand_acronyms(unknowns, content_preview,
                                                self.config.llm_call)
                ctx.acronyms.update(expanded)
            return ctx

        return self._run_local_mode(chunks)
