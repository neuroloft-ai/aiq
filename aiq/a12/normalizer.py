"""A12 — Content Normalizer

Converts non-text content (tables, figures, procedures) into searchable
text so RAG systems can retrieve information locked inside HTML structures.

What it does:
    For each table, figure, and procedure found in the document:
    1. Detects the element type (comparison table, process figure, etc.)
    2. Generates a SUMMARY (topic anchor for retrieval)
    3. Generates FULL CONTENT (the actual answer text)
    4. Tracks token changes (how many tokens were added/extracted)
    5. Keeps SOURCE REFERENCE for traceability

How it works:
    - Tables: auto-classified as comparison/lookup/data/log. Rule-based
      extraction for lookup/data. LLM for comparison/log (needs synthesis).
    - Figures: caption-based type detection (process/chart/structure).
      Rule-based from surrounding text. LLM/vision for richer extraction.
    - Procedures: ordered lists extracted as numbered steps. LLM optionally
      enriches with preconditions and outcomes from surrounding context.

Config:
    mode: "rule_only" | "rule_then_llm" | "llm_all" (default: "rule_only")
    llm_call: optional callable(prompt: str) -> str
    vision_call: optional callable(prompt: str, image_path: str) -> str
    domain_type: from A11 DomainContext (passed internally)
    image_dir: directory containing figure images for vision extraction

Config exposed to AIQConfig:
    normalize_mode  -> A12Config.mode       (default: "rule_only")
    image_dir       -> A12Config.image_dir  (default: "")
    (llm_call and vision_call wired from AIQConfig.llm_client / vision_client)
    (domain_type wired from A11 output)

Auto-detected (no user input needed):
    Table type (comparison/lookup/data/log) — from headers and headings
    Figure type (process/chart/structure) — from caption keywords
    Extractions themselves — the whole point of A12

    No user overrides for auto-detection. Users can edit individual
    extractions in the review UI, but that is a UI concern.

Input:  raw HTML (str), optional source_ref (str)
Output: ModuleOutput with .data["normalized_html"] and .findings = list[Extraction]

LLM required: No (rule_only mode handles tables and procedures).
    Figures benefit significantly from LLM/vision — rule-only may produce gaps.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import ModuleOutput, TokenChange


@dataclass
class A12Config:
    """Configuration for normalization.

    Modes:
      rule_only:     Rule-based extraction only. Gaps stay as gaps.
      rule_then_llm: Rule-based first, LLM for gaps/low confidence.
      llm_all:       LLM for all extractions (best quality, highest cost).
    """
    mode: str = "rule_only"  # "rule_only" | "rule_then_llm" | "llm_all"
    llm_call: Optional[Callable] = None
    vision_call: Optional[Callable] = None  # (prompt, image_path) -> str
    domain_type: str = ""  # from A11 DomainContext (e.g., "support")
    image_dir: str = ""    # directory with figure images


@dataclass
class Extraction:
    """One extracted element with summary + content."""
    element_type: str       # "table", "figure", "procedure"
    element_id: str         # "table_1", "figure_2", "procedure_3"
    source_ref: str         # "Table: Payment Processing Times" or position info
    summary: str            # topic anchor
    content: str            # full extracted text
    original_html: str      # original HTML for reference
    tokens_added: int       # approximate tokens added
    confidence: str = "high"  # "high" (good extraction), "low" (gap/inferred), "llm" (LLM generated)
    is_gap: bool = False    # True if no meaningful content could be extracted


# =====================================================================
# Table extraction
# =====================================================================

def _detect_table_type(headers: list[str], rows: list[list[str]],
                       heading: str) -> str:
    """Auto-detect table type from headers and heading.

    Order matters — more specific checks first.
    """
    h_lower = " ".join(headers).lower()
    heading_lower = (heading or "").lower()
    all_text = h_lower + " " + heading_lower

    # Log/history first (most specific — "recent cases", "support tickets")
    if any(w in all_text for w in ("recent", "history", "log", "ticket")):
        return "log"
    if "case" in h_lower and any(w in all_text for w in ("customer", "issue", "status")):
        return "log"
    # Comparison (plans, pricing, features)
    if any(w in all_text for w in ("comparison", "plan", "tier", "feature", "pricing")):
        return "comparison"
    # Lookup (error codes, FAQ)
    if any(w in all_text for w in ("error", "code", "meaning", "resolution", "faq")):
        return "lookup"
    # Data (field/value, details)
    if len(headers) == 2 and any(w in h_lower for w in ("field", "value", "detail")):
        return "data"
    return "data"


def _table_rows_to_text(headers: list[str], rows: list[list[str]]) -> str:
    """Format table rows as pipe-separated text for LLM prompt."""
    lines = []
    for row in rows:
        parts = []
        for i, v in enumerate(row):
            h = headers[i] if i < len(headers) else f"Col{i+1}"
            if v:
                parts.append(f"{h}: {v}")
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def _extract_table_rule(headers: list[str], rows: list[list[str]],
                        heading: str, table_type: str) -> tuple[str, str]:
    """Rule-based table extraction. Returns (summary, content).

    Adds a description line before the data based on table type.
    """
    # Summary
    if headers and heading:
        summary = f"{heading}: {', '.join(headers[:4]).lower()}"
    elif headers:
        summary = f"Table showing {', '.join(headers[:4]).lower()}"
    else:
        summary = heading or "Table"

    # Description line based on type
    n_rows = len(rows)
    if table_type == "lookup" and heading:
        desc = f"{heading} ({n_rows} entries):"
    elif table_type == "comparison" and heading:
        desc = f"{heading} — comparing {n_rows} options by {', '.join(headers[:3]).lower()}:"
    elif table_type == "data" and heading:
        desc = f"{heading} details:"
    elif table_type == "log" and heading:
        desc = f"{heading} — {n_rows} entries:"
    else:
        desc = f"{heading or 'Table'} ({n_rows} rows):"

    # Content — one sentence per row
    content_lines = [desc]
    for row in rows:
        if headers:
            parts = []
            for i, v in enumerate(row):
                h = headers[i] if i < len(headers) else ""
                if v and v.lower() not in ('', '-', 'n/a', 'none'):
                    parts.append(f"{h}: {v}" if h else v)
            if parts:
                content_lines.append(". ".join(parts) + ".")
        else:
            line = ", ".join(c for c in row if c)
            if line:
                content_lines.append(line + ".")

    return summary, "\n".join(content_lines)


def _build_table_prompt(headers: list[str], rows: list[list[str]],
                        heading: str, table_type: str,
                        context_before: str, domain_type: str) -> str:
    """Build LLM prompt for comparison and log tables."""
    rows_text = _table_rows_to_text(headers, rows)
    domain_label = domain_type or "general"

    return f"""ROLE: You are extracting searchable text from a table in a {domain_label} knowledge base for a RAG retrieval system.

TASK: Convert this table into text that can answer user questions about the data in it. The text will be stored as a chunk and retrieved when someone asks a related question.

TABLE HEADING: {heading or 'Unknown'}
TABLE TYPE: {table_type}
HEADERS: {', '.join(headers)}
DATA:
{rows_text}

RULES:
- For COMPARISON tables (plans, pricing): describe each option with all details, then state what makes each one different and best for what use case.
- For LOG tables (cases, history): summarize the types of entries and patterns, not individual rows. Note how many entries and what categories of issues appear.
- Include EVERY number, price, percentage, and timeframe
- Write complete sentences, not "Field: Value" pairs
- Start with one summary sentence

OUTPUT: Plain text. Start directly with content.

EXAMPLE (comparison table):
Three subscription plans are available:
- Starter ($29/mo): up to 5 users, 10 GB storage, email support only.
- Professional ($79/mo): up to 25 users, 100 GB storage, email + chat support, full API access.
- Enterprise (custom pricing): unlimited users and storage, 24/7 phone support with dedicated CSM.
Starter is best for small teams. Professional suits growing teams. Enterprise is for large organizations."""


def _extract_tables(html: str, mode: str = "rule_only",
                    llm_call: Optional[Callable] = None,
                    domain_type: str = "") -> list[tuple[str, Extraction]]:
    """Find tables in HTML, extract content based on table type.

    Lookup/data tables: always rule-based (structured facts work best).
    Comparison/log tables: LLM in Hybrid/API mode (needs synthesis).
    """
    results = []
    table_re = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)

    for idx, match in enumerate(table_re.finditer(html), 1):
        table_html = match.group(0)
        table_body = match.group(1)

        # Extract headers
        headers = re.findall(r'<th[^>]*>(.*?)</th>', table_body, re.IGNORECASE)
        headers = [_clean_cell(h) for h in headers]

        # Extract rows
        rows = []
        for tr_match in re.finditer(r'<tr[^>]*>(.*?)</tr>', table_body, re.DOTALL | re.IGNORECASE):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', tr_match.group(1), re.IGNORECASE)
            if cells:
                rows.append([_clean_cell(c) for c in cells])

        if not headers and not rows:
            continue

        heading = _find_preceding_heading(html, match.start())
        source_ref = f"Table: {heading}" if heading else f"Table {idx}"
        table_type = _detect_table_type(headers, rows, heading)

        # Decide: rule-based or LLM
        use_llm = (
            mode != "rule_only"
            and llm_call
            and table_type in ("comparison", "log")
        )

        if use_llm:
            # LLM for comparison/log tables
            context_before = _get_context_window(html, match.start(), "before", 200)
            ctx_clean = _clean_cell(context_before)
            prompt = _build_table_prompt(
                headers, rows, heading, table_type, ctx_clean, domain_type)
            try:
                llm_content = llm_call(prompt)
                if llm_content and llm_content.strip():
                    summary = heading or f"Table {idx}"
                    content = llm_content.strip()
                    confidence = "llm"
                else:
                    summary, content = _extract_table_rule(headers, rows, heading, table_type)
                    confidence = "high"
            except Exception:
                summary, content = _extract_table_rule(headers, rows, heading, table_type)
                confidence = "high"
        else:
            # Rule-based for lookup/data tables (and all tables in local mode)
            summary, content = _extract_table_rule(headers, rows, heading, table_type)
            confidence = "high"

        full_text = f"[Summary] {summary}.\n[Content] {content}"
        original_text_words = len(_clean_cell(table_html).split())
        tokens_added = max(0, len(full_text.split()) - original_text_words)

        results.append((table_html, Extraction(
            element_type="table",
            element_id=f"table_{idx}",
            source_ref=source_ref,
            summary=summary,
            content=content,
            original_html=table_html,
            tokens_added=tokens_added,
            confidence=confidence,
        )))

    return results


# =====================================================================
# Figure extraction
# =====================================================================

def _detect_figure_type(caption: str) -> str:
    """Auto-detect figure type from caption keywords."""
    c = caption.lower()
    if any(w in c for w in ("flow", "workflow", "process", "pipeline", "sequence")):
        return "process"
    if any(w in c for w in ("chart", "graph", "trend", "revenue", "volume", "metric")):
        return "chart"
    if any(w in c for w in ("diagram", "structure", "architecture", "comparison", "tier", "overview")):
        return "structure"
    return "unknown"


def _build_figure_prompt(fig_num: str, caption: str, fig_type: str,
                         has_image: bool, context_before: str,
                         context_after: str, domain_type: str) -> str:
    """Build the smart figure extraction prompt."""
    domain_label = domain_type if domain_type else "general"
    image_status = "available" if has_image else "missing"

    return f"""ROLE: You are a knowledge base analyst extracting searchable content from figures in a {domain_label} knowledge base.

TASK: Convert this figure into text that a customer support agent or end user can find through search and use as an answer.

FIGURE: [Figure {fig_num}: {caption}]
FIGURE TYPE: {fig_type}
IMAGE: {image_status}

SURROUNDING CONTEXT:
--- Before ---
{context_before}
--- After ---
{context_after}

EXTRACTION RULES:
- For PROCESS/WORKFLOW figures:
  * List every step as numbered items
  * Include: who does it, how long, what threshold triggers it
  * Capture ALL conditional branches: "If X then Y, otherwise Z"
  * Include the outcome of each path (approved → credit, rejected → denied)

- For CHART/DATA figures:
  * State exact numbers, percentages, timeframes
  * Describe the trend (increasing, decreasing, spike)
  * Note comparisons (before vs after, category vs category)

- For STRUCTURE/COMPARISON figures:
  * List each component/tier/category
  * State how they differ (features, limits, pricing)
  * Note which is default or recommended

QUALITY REQUIREMENTS:
- Every number mentioned in context MUST appear in your output
- Every actor/team mentioned MUST be named (not "they" or "the team")
- If a step has a time limit, include it (e.g., "within 4 hours")
- If a threshold triggers a branch, state the threshold (e.g., "if amount > $500")

CONFIDENCE:
- If context fully describes the figure → extract everything
- If context partially describes it → extract what you can, then: [INCOMPLETE: what's missing]
- If context says nothing useful about this figure → reply: NO_CONTEXT

OUTPUT: Plain text, no markdown headers. Start directly with the content.

EXAMPLE (process figure):
The refund workflow has two paths based on the refund amount:
1. Customer submits refund request via support portal.
2. Support agent reviews within 24 hours.
3. If amount exceeds $100: Manager reviews and approves or rejects.
   - If approved: refund is processed within 5 business days.
   - If rejected: customer is notified with denial reason.
4. If amount is $100 or less: refund is auto-approved.
5. Approved refunds are credited to the original payment method."""


def _find_figure_image(fig_num: str, image_dir: str) -> Optional[str]:
    """Find the image file for a figure number."""
    if not image_dir:
        return None
    from pathlib import Path
    img_dir = Path(image_dir)
    if not img_dir.exists():
        return None
    for pattern in [f"figure{fig_num}_*", f"figure_{fig_num}_*", f"fig{fig_num}_*"]:
        for f in img_dir.glob(pattern):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                return str(f)
    return None


def _build_vision_prompt(fig_num: str, caption: str, fig_type: str,
                         context_before: str, context_after: str,
                         domain_type: str) -> str:
    """Build prompt for vision model — sent alongside the image."""
    domain_label = domain_type or "general"
    return f"""You are extracting searchable text from a figure image in a {domain_label} knowledge base.

FIGURE: [Figure {fig_num}: {caption}]
FIGURE TYPE: {fig_type}

SURROUNDING TEXT:
Before: {context_before[:200]}
After: {context_after[:200]}

INSTRUCTIONS:
1. Describe EVERYTHING visible in the image — every box, label, arrow, number, condition
2. If it shows a PROCESS/WORKFLOW: list each step as numbered items. Include ALL conditional branches ("If X then Y, otherwise Z")
3. If it shows a CHART: state all data points, trends, axis labels, values
4. If it shows a STRUCTURE: describe all components and relationships
5. Include every number, percentage, timeframe, threshold, and team name visible
6. Write plain text that someone could use as an answer without seeing the image

OUTPUT: Start directly with the content. No markdown headers."""


def _extract_figures(html: str, mode: str = "rule_only",
                     llm_call: Optional[Callable] = None,
                     vision_call: Optional[Callable] = None,
                     domain_type: str = "",
                     image_dir: str = "") -> list[tuple[str, Extraction]]:
    """Find figure references, extract descriptions from context.

    Rule-based: extract from surrounding text using patterns.
    LLM: smart prompt with figure type detection and domain context.
    """
    results = []
    fig_re = re.compile(
        r'(?:<[^>]+>)?\s*\[Figure\s+(\d+):\s*(.*?)\]\s*(?:</[^>]+>)?',
        re.IGNORECASE | re.DOTALL,
    )

    for match in fig_re.finditer(html):
        fig_num = match.group(1)
        fig_caption = _clean_cell(match.group(2))
        fig_html = match.group(0)

        # Surrounding context
        context_before = _get_context_window(html, match.start(), direction="before", chars=400)
        context_after = _get_context_window(html, match.end(), direction="after", chars=600)
        ctx_before_clean = _clean_cell(context_before)
        ctx_after_clean = _clean_cell(context_after)

        # Source ref and metadata
        heading = _find_preceding_heading(html, match.start())
        source_ref = f"Figure {fig_num}: {fig_caption}" if fig_caption else f"Figure {fig_num}"
        has_image = "image not available" not in fig_caption.lower()
        fig_type = _detect_figure_type(fig_caption)

        # Caption keywords for rule-based relevance check
        caption_words = set(re.findall(r'\b[a-z]{3,}\b', fig_caption.lower()))
        caption_words -= {'image', 'not', 'available', 'figure'}

        extracted_desc = ""
        confidence = "high"
        is_gap = False

        # ── Rule-based extraction (Local and Hybrid first pass) ──
        if mode != "llm_all":
            # Look for explicit figure description phrases
            desc_patterns = [
                r'(?:as shown|the (?:diagram|chart|figure|workflow) (?:shows|illustrates|displays))[,:]?\s*(.*?)(?:\.|$)',
                r'(?:has|have)\s+(\d+)\s+(?:stages?|steps?|phases?)[:\s]+(.*?)(?:\.|$)',
            ]
            for pat in desc_patterns:
                m = re.search(pat, ctx_after_clean, re.IGNORECASE)
                if m:
                    candidate = m.group(0).strip()
                    candidate_words = set(re.findall(r'\b[a-z]{3,}\b', candidate.lower()))
                    if caption_words and candidate_words & caption_words:
                        extracted_desc = candidate
                        break
                    elif not caption_words:
                        extracted_desc = candidate
                        break

            # Fallback: relevant sentences from context
            if not extracted_desc and ctx_after_clean and caption_words:
                sents = re.split(r'(?<=[.!?])\s+', ctx_after_clean)
                relevant = [s.strip() for s in sents[:4]
                           if set(re.findall(r'\b[a-z]{3,}\b', s.lower())) & caption_words]
                if relevant:
                    extracted_desc = " ".join(relevant)
                    confidence = "low"

        # ── Vision / LLM extraction ──
        use_llm = (mode == "llm_all") or (mode == "rule_then_llm" and not extracted_desc)
        if use_llm:
            # Try 1: Vision model (if image available)
            img_path = _find_figure_image(fig_num, image_dir)
            if img_path and vision_call:
                try:
                    vision_prompt = _build_vision_prompt(
                        fig_num, fig_caption, fig_type,
                        ctx_before_clean[:200], ctx_after_clean[:200], domain_type,
                    )
                    vision_desc = vision_call(vision_prompt, img_path)
                    if vision_desc and vision_desc.strip():
                        extracted_desc = vision_desc.strip()
                        confidence = "llm"
                except Exception as _vis_err:
                    import logging
                    logging.getLogger("aiq.a12").warning("Vision call failed: %s", _vis_err)

            # Try 2: Text-only LLM (if vision didn't produce result)
            if not extracted_desc and llm_call:
                try:
                    prompt = _build_figure_prompt(
                        fig_num, fig_caption, fig_type, has_image,
                        ctx_before_clean[:300], ctx_after_clean[:500], domain_type,
                    )
                    llm_desc = llm_call(prompt)
                    if llm_desc:
                        llm_desc = llm_desc.strip()
                        if llm_desc == "NO_CONTEXT":
                            if not extracted_desc:
                                is_gap = True
                        elif "[INCOMPLETE:" in llm_desc:
                            extracted_desc = llm_desc
                            confidence = "llm"
                        else:
                            extracted_desc = llm_desc
                            confidence = "llm"
                except Exception as _llm_err:
                    import logging
                    logging.getLogger("aiq.a12").warning("LLM call failed: %s", _llm_err)

        # Final
        if not extracted_desc:
            extracted_desc = f"No description available for {fig_caption}."
            confidence = "low"
            is_gap = True
            summary = f"Figure {fig_num}: {fig_caption} (no description)"
        else:
            summary = f"Figure {fig_num}: {fig_caption}"

        content = extracted_desc
        if not has_image:
            content += " (Note: original image not available.)"

        full_text = f"[Summary] {summary}.\n[Content] {content}"
        original_text_words = len(_clean_cell(fig_html).split())
        tokens_added = max(0, len(full_text.split()) - original_text_words)

        results.append((fig_html, Extraction(
            element_type="figure",
            element_id=f"figure_{fig_num}",
            source_ref=source_ref,
            summary=summary,
            content=content,
            original_html=fig_html,
            tokens_added=tokens_added,
            confidence=confidence,
            is_gap=is_gap,
        )))

    return results


# =====================================================================
# Procedure extraction
# =====================================================================

def _build_procedure_prompt(heading: str, steps: list[str],
                            context_before: str, context_after: str,
                            domain_type: str) -> str:
    """Build LLM prompt for procedure enrichment."""
    domain_label = domain_type or "general"
    steps_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    return f"""ROLE: You are extracting searchable procedure text from a {domain_label} knowledge base for a RAG retrieval system.

TASK: Convert this procedure into complete, actionable instructions that answer "How do I do this?"

PROCEDURE HEADING: {heading or 'Unknown'}
STEPS:
{steps_text}

SURROUNDING CONTEXT:
--- Before ---
{context_before}
--- After ---
{context_after}

INSTRUCTIONS:
1. Keep ALL original steps -- do not skip or merge
2. Check context BEFORE the steps: does it mention any requirements or conditions the user must meet before starting? If yes, add a "Preconditions:" line before Step 1. If no, add exactly: [NO_PRECONDITIONS]
3. Check context AFTER the steps: does it mention what happens after completing the steps (processing time, confirmation, next steps)? If yes, add an "After completing:" line after the last step. If no, add exactly: [NO_OUTCOME]
4. Include specific timeframes, amounts, and contact info from context
5. Do NOT invent steps or details not in the text

OUTPUT FORMAT:
Summary line
[Preconditions: ... OR [NO_PRECONDITIONS]]
1. step
2. step
3. step
[After completing: ... OR [NO_OUTCOME]]"""


def _extract_procedures(html: str, mode: str = "rule_only",
                        llm_call: Optional[Callable] = None,
                        domain_type: str = "") -> list[tuple[str, Extraction]]:
    """Find ordered lists (procedures), extract with context.

    Rule-based: extract steps as numbered list.
    LLM (Hybrid/API): enrich with preconditions and outcomes from context.
    """
    results = []
    ol_re = re.compile(r'<ol[^>]*>(.*?)</ol>', re.DOTALL | re.IGNORECASE)

    for idx, match in enumerate(ol_re.finditer(html), 1):
        ol_html = match.group(0)
        steps = re.findall(r'<li[^>]*>(.*?)</li>', match.group(1), re.IGNORECASE)
        steps = [_clean_cell(s) for s in steps]

        if not steps:
            continue

        heading = _find_preceding_heading(html, match.start())
        context_before = _get_context_window(html, match.start(), "before", 300)
        context_after = _get_context_window(html, match.end(), "after", 300)
        ctx_before_clean = _clean_cell(context_before)
        ctx_after_clean = _clean_cell(context_after)

        source_ref = f"Procedure: {heading}" if heading else f"Procedure {idx}"

        # Summary
        if heading:
            summary = f"{len(steps)}-step procedure for {heading.lower()}"
        elif ctx_before_clean:
            summary = f"{len(steps)}-step procedure: {ctx_before_clean[:80]}"
        else:
            summary = f"Procedure with {len(steps)} steps"

        is_gap = False
        confidence = "high"

        # LLM enrichment
        use_llm = mode != "rule_only" and llm_call
        if use_llm:
            try:
                prompt = _build_procedure_prompt(
                    heading, steps, ctx_before_clean[:250],
                    ctx_after_clean[:250], domain_type,
                )
                llm_content = llm_call(prompt)
                if llm_content and llm_content.strip():
                    content = llm_content.strip()
                    confidence = "llm"
                    # Check for gap flags
                    if "[NO_PRECONDITIONS]" in content and "[NO_OUTCOME]" in content:
                        is_gap = True
                else:
                    # LLM failed — fall back to rule-based
                    content = " ".join(f"Step {i}: {s}." for i, s in enumerate(steps, 1))
            except Exception:
                content = " ".join(f"Step {i}: {s}." for i, s in enumerate(steps, 1))
        else:
            # Rule-based
            content = " ".join(f"Step {i}: {s}." for i, s in enumerate(steps, 1))

        full_text = f"[Summary] {summary}.\n[Content] {content}"
        original_text_words = len(_clean_cell(ol_html).split())
        tokens_added = max(0, len(full_text.split()) - original_text_words)

        results.append((ol_html, Extraction(
            element_type="procedure",
            element_id=f"procedure_{idx}",
            source_ref=source_ref,
            summary=summary,
            content=content,
            original_html=ol_html,
            tokens_added=tokens_added,
            confidence=confidence,
            is_gap=is_gap,
        )))

    return results


# =====================================================================
# Helpers
# =====================================================================

_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')


def _clean_cell(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = _TAG_RE.sub(' ', text)
    text = _WS_RE.sub(' ', text)
    return text.strip()


def _find_preceding_heading(html: str, pos: int) -> str:
    """Find the nearest heading before a position in the HTML."""
    heading_re = re.compile(r'<h[1-6]>(.*?)</h[1-6]>', re.IGNORECASE)
    last_heading = ""
    for m in heading_re.finditer(html):
        if m.end() <= pos:
            last_heading = _clean_cell(m.group(1))
        else:
            break
    return last_heading


def _get_context_window(html: str, pos: int, direction: str = "after", chars: int = 300) -> str:
    """Get text around a position for context."""
    if direction == "after":
        raw = html[pos:pos + chars]
    else:
        start = max(0, pos - chars)
        raw = html[start:pos]
    return raw


# =====================================================================
# Main normalizer
# =====================================================================

class Normalizer:
    """A12 — Convert non-text content to searchable text."""

    def __init__(self, config: Optional[A12Config] = None):
        self.config = config or A12Config()

    def run(self, html: str, source_ref: str = "") -> ModuleOutput:
        """Normalize document content.

        Args:
            html: raw document HTML
            source_ref: document source reference

        Returns:
            ModuleOutput with:
              - normalized text in extra["normalized_html"]
              - extractions list in extra["extractions"]
              - token changes tracking
        """
        t0 = time.perf_counter()
        all_extractions: list[Extraction] = []
        token_changes: list[TokenChange] = []
        normalized = html

        # 1. Extract tables
        table_results = _extract_tables(
            html, self.config.mode, self.config.llm_call, self.config.domain_type)
        for original_html, extraction in table_results:
            replacement = (
                f'\n<p class="aiq-extracted" data-source="{extraction.element_id}">'
                f'{extraction.summary}. {extraction.content}</p>\n'
            )
            normalized = normalized.replace(original_html, replacement, 1)
            all_extractions.append(extraction)
            token_changes.append(TokenChange(
                change_type="added",
                reason="table_extraction",
                token_count=extraction.tokens_added,
                module="A12",
                detail=extraction.source_ref,
            ))

        # 2. Extract figures (search original HTML for context, not the modified normalized)
        figure_results = _extract_figures(
            html, self.config.mode, self.config.llm_call,
            self.config.vision_call, self.config.domain_type, self.config.image_dir)
        for original_html, extraction in figure_results:
            replacement = (
                f'\n<p class="aiq-extracted" data-source="{extraction.element_id}">'
                f'{extraction.summary}. {extraction.content}</p>\n'
            )
            normalized = normalized.replace(original_html, replacement, 1)
            all_extractions.append(extraction)
            token_changes.append(TokenChange(
                change_type="added",
                reason="figure_extraction",
                token_count=extraction.tokens_added,
                module="A12",
                detail=extraction.source_ref,
            ))

        # 3. Extract procedures
        procedure_results = _extract_procedures(
            html, self.config.mode, self.config.llm_call, self.config.domain_type)
        for original_html, extraction in procedure_results:
            replacement = (
                f'\n<p class="aiq-extracted" data-source="{extraction.element_id}">'
                f'{extraction.summary}. {extraction.content}</p>\n'
            )
            normalized = normalized.replace(original_html, replacement, 1)
            all_extractions.append(extraction)
            token_changes.append(TokenChange(
                change_type="added",
                reason="procedure_extraction",
                token_count=extraction.tokens_added,
                module="A12",
                detail=extraction.source_ref,
            ))

        detected = len(all_extractions)
        gaps = sum(1 for e in all_extractions if e.is_gap)
        resolved = detected - gaps

        words_in = len(re.sub(r'<[^>]+>', ' ', html).split())
        words_out = len(re.sub(r'<[^>]+>', ' ', normalized).split())

        return ModuleOutput(
            module_id="A12",
            module_name="Normalize",
            detected=detected,
            resolved=resolved,
            remaining=gaps,
            words_in=words_in,
            words_out=words_out,
            token_changes=token_changes,
            elapsed_seconds=time.perf_counter() - t0,
            findings=all_extractions,
            data={"normalized_html": normalized},
        )
