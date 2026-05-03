"""A22 — Metadata Enrichment

Extracts temporal and version data from chunk content, merges with
source-level metadata (author, last_modified, etc.), and optionally
flags stale content.

What it does:
    1. Extracts dates from content (Month Year, Q1 2024, ISO dates, dated phrases)
    2. Extracts version references (v3.2, Version 4.0)
    3. Reads source metadata already on chunk.metadata (from document input)
    4. Calculates page age from last_modified
    5. Optionally flags chunks as STALE if page exceeds age threshold

What it adds to chunk.metadata:
    a22_dates:               list of date mentions with age in months
    a22_versions:            list of version strings found in content
    a22_dated_phrases:       list of temporal phrases ("updated January 2026")
    a22_page_age_months:     int — months since last page edit
    a22_newest_content_date: dict — the most recent date mentioned in content

Consumed by:
    A32 (Consistency): uses dates for contradiction winner suggestion
    A43 (Metrics): freshness reporting

Config:
    flag_stale: whether to tag old chunks as STALE (default: False)
    stale_months: age threshold in months (default: 18)
    reference_date: date to calculate age against (default: now)
    min_year / max_year: year range for content date extraction (default: 2015-2030)

Config exposed to AIQConfig:
    freshness_threshold_days -> A22Config.flag_stale + stale_months (default: 180 days, 0=disabled)
    (source metadata like author, last_modified comes from document input, not config)

Auto-detected (no user input needed):
    Content dates — regex patterns for Month Year, Q1 2024, ISO dates
    Version references — regex for v3.2, Version 4.0
    Page age — calculated from last_modified in chunk.metadata

    Source metadata (author, last_modified, status, labels) is NOT auto-detected
    by A22 — it must be provided at document input time. A22 reads and uses it.

Input:  list[Chunk] (with optional metadata from document input)
Output: ModuleOutput with enrichment counts in .data

LLM required: No. Fully rule-based.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from aiq.core.types import Chunk, ChunkTag, ModuleOutput, TokenChange


# =====================================================================
# Config
# =====================================================================

@dataclass
class A22Config:
    """Configuration for metadata enrichment."""
    # Stale flagging (optional, user-driven)
    flag_stale: bool = False
    stale_months: int = 18
    # Reference date for age calculation
    reference_date: datetime = field(default_factory=datetime.now)
    # Year range for content date extraction
    min_year: int = 2015
    max_year: int = 2030


# =====================================================================
# Date extraction patterns
# =====================================================================

_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(\d{1,2},?\s+)?(\d{4})\b",
    re.IGNORECASE,
)

_QUARTER_YEAR_RE = re.compile(r"\bQ([1-4])\s+(\d{4})\b")

_ISO_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")

_DATED_PHRASE_RE = re.compile(
    r"\b(?:updated|as of|effective|since|from|dated|last (?:edited|modified))\s+"
    r"((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(?:\d{1,2},?\s+)?\d{4})\b",
    re.IGNORECASE,
)

_VERSION_RE = re.compile(r"\b[Vv](?:ersion)?\s*(\d+(?:\.\d+)*)\b")


@dataclass
class DateMention:
    """A date found in content."""
    year: int
    month: int
    raw_text: str
    date_type: str  # "month_year", "quarter", "iso", "dated_phrase"
    age_months: int = 0


def _extract_dates(text: str, config: A22Config) -> list[DateMention]:
    """Extract all date mentions from text."""
    dates = []

    for m in _MONTH_YEAR_RE.finditer(text):
        month_str = m.group(1).lower().rstrip('.')
        month = _MONTH_MAP.get(month_str, 0)
        year = int(m.group(3))
        if config.min_year <= year <= config.max_year:
            age = (config.reference_date.year - year) * 12 + (config.reference_date.month - month)
            dates.append(DateMention(year=year, month=month, raw_text=m.group(),
                                     date_type="month_year", age_months=age))

    for m in _QUARTER_YEAR_RE.finditer(text):
        quarter = int(m.group(1))
        year = int(m.group(2))
        month = (quarter - 1) * 3 + 1
        if config.min_year <= year <= config.max_year:
            age = (config.reference_date.year - year) * 12 + (config.reference_date.month - month)
            dates.append(DateMention(year=year, month=month, raw_text=m.group(),
                                     date_type="quarter", age_months=age))

    for m in _ISO_DATE_RE.finditer(text):
        year, month = int(m.group(1)), int(m.group(2))
        if config.min_year <= year <= config.max_year and 1 <= month <= 12:
            age = (config.reference_date.year - year) * 12 + (config.reference_date.month - month)
            dates.append(DateMention(year=year, month=month, raw_text=m.group(),
                                     date_type="iso", age_months=age))

    return dates


# =====================================================================
# Page metadata parsing
# =====================================================================

def _parse_page_date(last_modified: str) -> Optional[datetime]:
    """Parse Confluence last_modified ISO datetime."""
    if not last_modified:
        return None
    try:
        date_part = last_modified.split("T")[0] if "T" in last_modified else last_modified[:10]
        return datetime.strptime(date_part, "%Y-%m-%d")
    except (ValueError, IndexError):
        return None


# =====================================================================
# Main enricher
# =====================================================================

class MetadataEnricher:
    """A22 — Extract dates, versions, entities and store on chunk metadata."""

    def __init__(self, config: Optional[A22Config] = None):
        self.config = config or A22Config()

    def run(self, chunks: list[Chunk]) -> ModuleOutput:
        """Enrich chunks with metadata.

        Adds to each chunk.metadata:
          - "a22_dates": list of DateMention dicts
          - "a22_versions": list of version strings
          - "a22_dated_phrases": list of dated phrases
          - "a22_page_age_months": int (if page metadata available)
          - "a22_newest_content_date": dict (the most recent date mentioned)

        If flag_stale is enabled, chunks from old pages get tagged.

        Returns:
            ModuleOutput with enrichment counts
        """
        t0 = time.perf_counter()
        words_in = sum(c.words for c in chunks)

        enriched = 0
        stale_flagged = 0
        total_dates = 0
        total_versions = 0

        for chunk in chunks:
            content = chunk.content
            metadata = chunk.metadata

            # Extract content dates
            dates = _extract_dates(content, self.config)
            if dates:
                metadata["a22_dates"] = [
                    {"year": d.year, "month": d.month, "raw": d.raw_text,
                     "type": d.date_type, "age_months": d.age_months}
                    for d in dates
                ]
                # Track newest content date for A32 contradiction resolution
                newest = min(dates, key=lambda d: d.age_months)
                metadata["a22_newest_content_date"] = {
                    "year": newest.year, "month": newest.month,
                    "raw": newest.raw_text, "age_months": newest.age_months,
                }
                total_dates += len(dates)

            # Extract versions
            versions = [m.group(1) for m in _VERSION_RE.finditer(content)]
            if versions:
                metadata["a22_versions"] = versions
                total_versions += len(versions)

            # Extract dated phrases
            dated_phrases = [m.group() for m in _DATED_PHRASE_RE.finditer(content)]
            if dated_phrases:
                metadata["a22_dated_phrases"] = dated_phrases

            # Page metadata (Confluence last_modified)
            last_mod_str = metadata.get("last_modified", "")
            if last_mod_str:
                last_mod = _parse_page_date(last_mod_str)
                if last_mod:
                    age = (self.config.reference_date.year - last_mod.year) * 12 + (
                        self.config.reference_date.month - last_mod.month
                    )
                    metadata["a22_page_age_months"] = age

                    # Optional stale flagging
                    if self.config.flag_stale and age > self.config.stale_months:
                        if chunk.tag == ChunkTag.CONTENT:
                            chunk.tag = ChunkTag.STALE
                            chunk.tag_reason = (
                                f"Page last edited {age} months ago "
                                f"(threshold: {self.config.stale_months} months)"
                            )
                            chunk.tag_module = "A22"
                            stale_flagged += 1

            if dates or versions or dated_phrases or last_mod_str:
                enriched += 1

        detected = enriched
        resolved = enriched  # enrichment is always "resolved" (data added)

        return ModuleOutput(
            module_id="A22",
            module_name="Metadata Enrichment",
            detected=detected,
            resolved=resolved,
            remaining=stale_flagged,  # stale items need user attention
            words_in=words_in,
            words_out=words_in,  # A22 doesn't change content
            elapsed_seconds=time.perf_counter() - t0,
            data={
                "total_dates": total_dates,
                "total_versions": total_versions,
                "stale_flagged": stale_flagged,
                "chunks_enriched": enriched,
            },
        )
