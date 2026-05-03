"""A13 — Structure & Headings

Ensures every section has a descriptive heading that helps retrieval.
Headings are critical metadata — they provide topic signals that help
retrieval engines match queries to the right chunk.

What it does:
    1. Parses existing headings from HTML (h1-h6 tags)
    2. Detects orphaned content (sections with no heading)
    3. Detects generic headings ("Notes", "Other", "Misc", "Section 1")
    4. Generates descriptive headings from content to replace missing/generic ones

How it works:
    - Rule-based heading generation uses 4 strategies:
      1. Subject-verb patterns ("Billing handles invoices")
      2. About/regarding patterns ("about refund processing")
      3. Most frequent topic words across content
      4. Fallback: first meaningful words
    - LLM mode generates short (3-6 word) descriptive headings
    - Generic detection uses a curated list of non-descriptive terms

Config:
    mode: "rule_only" | "rule_then_llm" | "llm_all" (default: "rule_only")
    llm_call: optional callable(prompt: str) -> str
    min_section_words: minimum words to consider a section (default: 10)

Config exposed to AIQConfig:
    structure_mode       -> A13Config.mode              (default: "rule_only")
    min_section_words    -> A13Config.min_section_words  (default: 10)
    (llm_call wired from AIQConfig.llm_client)

Auto-detected (no user input needed):
    Orphaned sections — content without any heading above it
    Generic headings — matched against curated list of non-descriptive terms
    Generated headings — produced from content analysis

    No user overrides for auto-detection. Users can edit generated headings
    in the review UI before they are applied.

Input:  HTML (str) from A12 normalized output, optional source_ref (str)
Output: ModuleOutput with .data["sections"] = list[Section], .findings = list[Section]

LLM required: No (rule_only mode generates headings from content).
    LLM produces better headings but rule-based is functional.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from aiq.core.types import ModuleOutput, TokenChange


# =====================================================================
# Config
# =====================================================================

@dataclass
class A13Config:
    """Configuration for structure detection and fixing."""
    mode: str = "rule_only"  # "rule_only" | "rule_then_llm" | "llm_all"
    llm_call: Optional[Callable] = None
    # Minimum content words to be considered a section (skip tiny fragments)
    min_section_words: int = 10


# =====================================================================
# Section dataclass
# =====================================================================

@dataclass
class Section:
    """A document section with heading, level, and content."""
    section_id: str
    heading: str
    heading_level: int          # 1-6, or 0 if no heading detected
    content: str
    words: int
    # Structure fix tracking
    heading_source: str = "original"  # "original", "generated_rule", "generated_llm", "replaced"
    original_heading: str = ""         # if heading was replaced, keep original
    is_orphan: bool = False            # had no heading before A13
    is_generic: bool = False           # heading was generic ("Notes", "Other")


# =====================================================================
# Generic heading detection
# =====================================================================

_GENERIC_HEADINGS = re.compile(
    r'^(?:other|misc(?:ellaneous)?|notes?|stuff|random|untitled|general|'
    r'section\s*\d+|part\s*\d+|additional\s*(?:info(?:rmation)?)?|'
    r'more\s*(?:info(?:rmation)?)?|overview|details|content|'
    r'appendix|extra|various|update|updates)\s*$',
    re.IGNORECASE,
)


def _is_generic_heading(heading: str) -> bool:
    """Check if a heading is too generic to be useful for retrieval."""
    cleaned = heading.strip().rstrip(':').strip()
    if not cleaned:
        return True
    if _GENERIC_HEADINGS.match(cleaned):
        return True
    # Single word that's not descriptive
    if len(cleaned.split()) == 1 and len(cleaned) < 5:
        return True
    return False


# =====================================================================
# Rule-based heading generation
# =====================================================================

def _generate_heading_rule(content: str) -> str:
    """Generate a heading from content using rule-based extraction.

    Strategy:
      1. Look for subject pattern: "[Entity] handles/manages/processes X"
      2. Look for "about/regarding" patterns
      3. Find the most frequent topic words across all sentences
      4. Fallback: first meaningful words
    """
    # Clean HTML
    clean = re.sub(r'<[^>]+>', ' ', content)
    clean = re.sub(r'\s+', ' ', clean).strip()

    if not clean:
        return "Untitled Section"

    # Get first 2 sentences
    sents = re.split(r'(?<=[.!?])\s+', clean)
    first_sents = ' '.join(sents[:2]) if sents else clean[:150]

    # Strategy 1: "[Entity] handles/manages/processes [topic]"
    subject_match = re.search(
        r'(\w+)\s+(?:handles?|manages?|processes?|covers?|provides?|supports?)\s+'
        r'(?:all\s+)?(.{5,40}?)(?:\s+(?:for|across|and|including)|[.,;]|$)',
        first_sents, re.IGNORECASE,
    )
    if subject_match:
        topic = subject_match.group(2).strip().rstrip('.,;:')
        if topic and len(topic.split()) <= 5:
            return topic[0].upper() + topic[1:]

    # Strategy 2: "about/regarding/for" patterns
    about_match = re.search(
        r'(?:about|regarding|for|covers?|describes?|explains?)\s+(.{10,60}?)(?:[.,;]|$)',
        first_sents, re.IGNORECASE,
    )
    if about_match:
        phrase = about_match.group(1).strip().rstrip('.')
        if phrase:
            return phrase[0].upper() + phrase[1:]

    # Strategy 3: most frequent topic words across content
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
        'for', 'to', 'in', 'on', 'at', 'by', 'with', 'from', 'of',
        'all', 'our', 'we', 'they', 'you', 'your', 'has', 'have',
        'can', 'will', 'may', 'should', 'must', 'not', 'any', 'some',
        'also', 'out', 'via', 'per', 'each', 'every', 'more', 'most',
        'been', 'being', 'into', 'than', 'then', 'when', 'where',
        'which', 'what', 'how', 'who', 'but', 'if', 'them', 'their',
        'there', 'here', 'very', 'just', 'make', 'take', 'get',
        'contact', 'check', 'see', 'need', 'use', 'available',
    }
    words_lower = re.findall(r'\b[a-z]{3,}\b', clean.lower())
    word_freq = {}
    for w in words_lower:
        if w not in stop_words:
            word_freq[w] = word_freq.get(w, 0) + 1

    if word_freq:
        # Top 2-3 most frequent words
        top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:3]
        heading = ' '.join(w.capitalize() for w in top_words)
        return heading

    # Strategy 4: fallback to first meaningful words
    words = clean.split()
    meaningful = [w for w in words[:15] if w.lower().rstrip('.,;:') not in stop_words and len(w) > 2]
    if meaningful:
        heading = ' '.join(meaningful[:4]).rstrip('.,;:')
        return heading[0].upper() + heading[1:] if heading else "Untitled Section"

    return "Untitled Section"


def _generate_heading_llm(content: str, llm_call: Callable) -> str:
    """Generate a heading using LLM."""
    clean = re.sub(r'<[^>]+>', ' ', content)
    clean = re.sub(r'\s+', ' ', clean).strip()[:300]

    prompt = (
        f"Generate a short, descriptive heading (3-6 words) for this content. "
        f"The heading should help someone searching for this information find it.\n\n"
        f"Content:\n{clean}\n\n"
        f"Return ONLY the heading text, nothing else."
    )
    try:
        result = llm_call(prompt)
        # Clean up LLM response
        result = result.strip().strip('"').strip("'").strip('#').strip()
        if result and 2 <= len(result.split()) <= 8:
            return result
    except Exception:
        pass
    return ""


# =====================================================================
# HTML parsing
# =====================================================================

_HEADING_RE = re.compile(r'<h([1-6])>(.*?)</h\1>', re.IGNORECASE | re.DOTALL)


def _parse_sections(html: str) -> list[Section]:
    """Parse HTML into sections based on heading tags."""
    # Split at heading boundaries
    parts = re.split(r'(<h[1-6]>.*?</h[1-6]>)', html, flags=re.IGNORECASE)

    sections = []
    current_heading = ""
    current_level = 0
    current_content = ""
    section_idx = 0

    for part in parts:
        heading_match = _HEADING_RE.match(part)
        if heading_match:
            # Save previous section if it has content
            if current_content.strip():
                clean_content = re.sub(r'<[^>]+>', ' ', current_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                words = len(clean_content.split())
                if words > 0:
                    sections.append(Section(
                        section_id=f"sec_{section_idx}",
                        heading=current_heading,
                        heading_level=current_level,
                        content=current_content.strip(),
                        words=words,
                        is_orphan=(current_level == 0 and not current_heading),
                    ))
                    section_idx += 1

            # Start new section
            current_level = int(heading_match.group(1))
            current_heading = re.sub(r'<[^>]+>', '', heading_match.group(2)).strip()
            current_content = ""
        else:
            current_content += part

    # Don't forget last section
    if current_content.strip():
        clean_content = re.sub(r'<[^>]+>', ' ', current_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        words = len(clean_content.split())
        if words > 0:
            sections.append(Section(
                section_id=f"sec_{section_idx}",
                heading=current_heading,
                heading_level=current_level,
                content=current_content.strip(),
                words=words,
                is_orphan=(current_level == 0 and not current_heading),
            ))

    return sections


# =====================================================================
# Main structurer
# =====================================================================

class Structurer:
    """A13 — Heading detection, generation, and hierarchy."""

    def __init__(self, config: Optional[A13Config] = None):
        self.config = config or A13Config()

    def run(self, html: str, source_ref: str = "") -> ModuleOutput:
        """Parse structure, fix missing/generic headings.

        Args:
            html: document HTML (from A12 normalized output)
            source_ref: document source reference

        Returns:
            ModuleOutput with sections in .findings and fix counts
        """
        t0 = time.perf_counter()
        token_changes: list[TokenChange] = []

        # Parse into sections
        sections = _parse_sections(html)

        detected = 0
        resolved = 0

        for section in sections:
            if section.words < self.config.min_section_words:
                continue

            needs_fix = False

            # Check 1: orphaned content (no heading)
            if section.is_orphan:
                detected += 1
                needs_fix = True

            # Check 2: generic heading
            elif _is_generic_heading(section.heading):
                section.is_generic = True
                section.original_heading = section.heading
                detected += 1
                needs_fix = True

            if not needs_fix:
                continue

            # Fix: generate heading
            new_heading = ""

            if self.config.mode == "llm_all" and self.config.llm_call:
                new_heading = _generate_heading_llm(section.content, self.config.llm_call)
                if new_heading:
                    section.heading_source = "generated_llm"

            if not new_heading and self.config.mode != "llm_all":
                # Rule-based
                new_heading = _generate_heading_rule(section.content)
                if new_heading and new_heading != "Untitled Section":
                    section.heading_source = "generated_rule"
                elif self.config.mode == "rule_then_llm" and self.config.llm_call:
                    # Rule failed, try LLM
                    llm_heading = _generate_heading_llm(section.content, self.config.llm_call)
                    if llm_heading:
                        new_heading = llm_heading
                        section.heading_source = "generated_llm"

            if new_heading and new_heading != "Untitled Section":
                if section.is_generic:
                    section.heading_source = "replaced"
                section.heading = new_heading
                if section.heading_level == 0:
                    section.heading_level = 2  # default level for generated headings
                resolved += 1

                token_changes.append(TokenChange(
                    change_type="added",
                    reason="heading_generation",
                    token_count=len(new_heading.split()),
                    module="A13",
                    detail=f"{'Replaced' if section.is_generic else 'Generated'}: {new_heading}",
                ))

        words_in = sum(s.words for s in sections)
        words_added = sum(tc.token_count for tc in token_changes)
        words_out = words_in + words_added

        return ModuleOutput(
            module_id="A13",
            module_name="Structure",
            detected=detected,
            resolved=resolved,
            remaining=detected - resolved,
            words_in=words_in,
            words_out=words_out,
            findings=sections,
            token_changes=token_changes,
            elapsed_seconds=time.perf_counter() - t0,
            data={"sections": sections},
        )
