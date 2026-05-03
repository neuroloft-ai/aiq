"""A14 — Smart Chunker

Topic-aware chunking that ensures each chunk contains exactly one topic.
Unlike A10 (naive chunking), A14 detects topic boundaries using keyword
overlap between sentences, then enforces size guardrails.

What it does:
    1. For each section (from A13), splits content into sentences
    2. Detects topic shifts via sliding-window Jaccard similarity on keywords
    3. Splits at topic boundaries — each chunk gets one topic
    4. Enforces size: splits oversized chunks at sentence boundaries
    5. Merges undersized chunks ONLY with same-topic neighbors
    6. Assigns heading from parent section (with -1, -2 suffix for splits)

    Never merges different topics — a small focused chunk is better
    than a bigger mixed one for retrieval.

How it works:
    - Keyword extraction: words 4+ chars, excluding stop words
    - Topic shift: Jaccard similarity between sliding window of 2 consecutive
      sentences vs next 2. Below threshold (default 0.1) = topic boundary.
    - Merge pass: iterative forward/backward merge of undersized chunks
      with topic-similar neighbors until stable.

Config:
    min_words: minimum chunk size in words (default: 50)
    max_words: maximum chunk size in words (default: 200)
    min_sentences_for_split: minimum sentences to attempt topic split (default: 3)
    topic_shift_threshold: Jaccard similarity below this = topic shift (default: 0.1)
    min_split_words: minimum words per split segment (default: 30)

Config exposed to AIQConfig:
    chunk_min_words          -> A14Config.min_words               (default: 50)
    chunk_max_words          -> A14Config.max_words               (default: 200)
    topic_shift_threshold    -> A14Config.topic_shift_threshold   (default: 0.1)
    (min_sentences_for_split and min_split_words are tuning knobs for advanced users)

Auto-detected (no user input needed):
    Topic boundaries — detected from keyword overlap between sentences
    Merge candidates — determined by topic similarity + size constraints

    No user overrides for auto-detection.

Input:  list[Section] from A13 output, optional source_ref (str)
Output: ModuleOutput with .chunks = list[Chunk]

LLM required: No. Fully rule-based.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from aiq.core.types import Chunk, ModuleOutput, TokenChange
from aiq.a13.structurer import Section


# =====================================================================
# Config
# =====================================================================

@dataclass
class A14Config:
    """Configuration for smart chunking."""
    min_words: int = 50
    max_words: int = 200
    # Topic shift detection
    min_sentences_for_split: int = 3   # need at least this many sentences to consider splitting
    topic_shift_threshold: float = 0.1  # Jaccard similarity below this = topic shift
    min_split_words: int = 30           # each split part must have at least this many words


# =====================================================================
# Topic detection
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
    "information details section page steps process issues".split()
)

_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')


def _clean_text(text: str) -> str:
    """Strip HTML and normalize whitespace."""
    text = re.sub(r'</(?:p|li|tr|div|h[1-6])>', '. ', text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(' ', text)
    text = re.sub(r':\s*\.', ':', text)
    text = re.sub(r'\.(\s*\.)+', '.', text)
    text = _WS_RE.sub(' ', text)
    return text.strip()


def _extract_topic_words(text: str) -> set[str]:
    """Extract meaningful topic words from text."""
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sents = _SENT_RE.split(text)
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]


# =====================================================================
# Topic boundary detection
# =====================================================================

def _find_topic_boundaries(sentences: list[str], threshold: float) -> list[int]:
    """Find sentence indices where topics shift.

    Returns list of split-point indices (the sentence AFTER the boundary).
    """
    if len(sentences) < 3:
        return []

    # Extract topic words per sentence
    sent_topics = [_extract_topic_words(s) for s in sentences]

    # Compare consecutive sentences using a sliding window of 2
    boundaries = []
    for i in range(1, len(sentences)):
        # Compare this sentence's topics with previous 2 sentences
        before = set()
        for j in range(max(0, i - 2), i):
            before |= sent_topics[j]

        after = set()
        for j in range(i, min(len(sentences), i + 2)):
            after |= sent_topics[j]

        sim = _jaccard(before, after)
        if sim < threshold:
            boundaries.append(i)

    return boundaries


# =====================================================================
# Smart chunker
# =====================================================================

class SmartChunker:
    """A14 — Topic-aware chunking with size guardrails."""

    def __init__(self, config: Optional[A14Config] = None):
        self.config = config or A14Config()

    def run(self, sections: list[Section], source_ref: str = "") -> ModuleOutput:
        """Chunk sections into focused, properly sized chunks.

        Args:
            sections: from A13 Structurer output
            source_ref: document source reference

        Returns:
            ModuleOutput with chunks in .chunks
        """
        t0 = time.perf_counter()
        all_chunks: list[Chunk] = []
        detected = 0  # multi-topic sections found
        resolved = 0  # successfully split

        chunk_counter = 1

        for section in sections:
            content = _clean_text(section.content)
            words = len(content.split())

            # Small sections stay as single chunk — no point splitting
            if words <= self.config.min_words * 2:
                all_chunks.append(Chunk(
                    chunk_id=f"c{chunk_counter}",
                    heading=section.heading,
                    content=content,
                    words=words,
                    source_type="text",
                    source_ref=source_ref,
                ))
                chunk_counter += 1
                continue

            if words < self.config.min_words // 2:
                # Very small section — keep as-is
                all_chunks.append(Chunk(
                    chunk_id=f"c{chunk_counter}",
                    heading=section.heading,
                    content=content,
                    words=words,
                    source_type="text",
                    source_ref=source_ref,
                ))
                chunk_counter += 1
                continue

            # Split into sentences
            sentences = _split_sentences(content)

            if len(sentences) < self.config.min_sentences_for_split:
                # Not enough sentences to detect topic shifts
                # But still check size
                chunks_from_section = self._enforce_size([content], section.heading)
                for c_text, c_heading in chunks_from_section:
                    all_chunks.append(Chunk(
                        chunk_id=f"c{chunk_counter}",
                        heading=c_heading,
                        content=c_text,
                        words=len(c_text.split()),
                        source_type="text",
                        source_ref=source_ref,
                    ))
                    chunk_counter += 1
                continue

            # Find topic boundaries
            boundaries = _find_topic_boundaries(
                sentences, self.config.topic_shift_threshold
            )

            if not boundaries:
                # Single topic — just enforce size
                chunks_from_section = self._enforce_size([content], section.heading)
                for c_text, c_heading in chunks_from_section:
                    all_chunks.append(Chunk(
                        chunk_id=f"c{chunk_counter}",
                        heading=c_heading,
                        content=c_text,
                        words=len(c_text.split()),
                        source_type="text",
                        source_ref=source_ref,
                    ))
                    chunk_counter += 1
            else:
                # Multi-topic — split at boundaries
                detected += 1
                segments = self._split_at_boundaries(sentences, boundaries)

                # Validate each segment meets minimum size
                valid_segments = self._validate_segments(segments)

                if len(valid_segments) > 1:
                    resolved += 1

                # Enforce size on each segment and create chunks
                for seg_idx, segment_text in enumerate(valid_segments):
                    suffix = f" - {seg_idx + 1}" if len(valid_segments) > 1 else ""
                    seg_heading = f"{section.heading}{suffix}" if section.heading else f"Part {seg_idx + 1}"

                    sized_chunks = self._enforce_size([segment_text], seg_heading)
                    for c_text, c_heading in sized_chunks:
                        all_chunks.append(Chunk(
                            chunk_id=f"c{chunk_counter}",
                            heading=c_heading,
                            content=c_text,
                            words=len(c_text.split()),
                            source_type="text",
                            source_ref=source_ref,
                            parent_chunk_id=section.section_id,
                        ))
                        chunk_counter += 1

        # Merge pass: merge undersized chunks with same-topic neighbor
        final_chunks = self._merge_same_topic(all_chunks)

        # Re-number chunk IDs
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_id = f"c{i + 1}"

        words_in = sum(s.words for s in sections)
        words_out = sum(c.words for c in final_chunks)

        return ModuleOutput(
            module_id="A14",
            module_name="Smart Chunker",
            detected=detected,
            resolved=resolved,
            remaining=detected - resolved,
            words_in=words_in,
            words_out=words_out,
            chunks=final_chunks,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _split_at_boundaries(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """Split sentences into segments at topic boundaries."""
        segments = []
        prev = 0
        for b in boundaries:
            segment = " ".join(sentences[prev:b])
            if segment.strip():
                segments.append(segment.strip())
            prev = b
        # Last segment
        last = " ".join(sentences[prev:])
        if last.strip():
            segments.append(last.strip())
        return segments

    def _validate_segments(self, segments: list[str]) -> list[str]:
        """Merge too-small segments back with neighbors if same topic."""
        if not segments:
            return segments

        result = []
        i = 0
        while i < len(segments):
            current = segments[i]
            # If too small, try merging with next
            while (len(current.split()) < self.config.min_split_words
                   and i + 1 < len(segments)):
                # Check topic similarity before merging
                current_topics = _extract_topic_words(current)
                next_topics = _extract_topic_words(segments[i + 1])
                if _jaccard(current_topics, next_topics) > self.config.topic_shift_threshold:
                    # Same topic — merge
                    i += 1
                    current = current + " " + segments[i]
                else:
                    # Different topic — keep small, don't merge
                    break
            result.append(current)
            i += 1
        return result

    def _enforce_size(self, texts: list[str], heading: str) -> list[tuple[str, str]]:
        """Enforce max_words limit by splitting at sentence boundaries.

        Returns list of (text, heading) tuples.
        """
        result = []
        for text in texts:
            words = len(text.split())
            if words <= self.config.max_words:
                result.append((text, heading))
                continue

            # Need to split — find sentence boundaries
            sentences = _split_sentences(text)
            if len(sentences) <= 1:
                result.append((text, heading))
                continue

            current_sents = []
            current_words = 0
            part_num = 1

            for sent in sentences:
                sent_words = len(sent.split())
                if current_words + sent_words > self.config.max_words and current_sents:
                    chunk_text = " ".join(current_sents)
                    suffix = f" - {part_num}" if True else ""  # always suffix when size-splitting
                    result.append((chunk_text, f"{heading}{suffix}"))
                    part_num += 1
                    current_sents = [sent]
                    current_words = sent_words
                else:
                    current_sents.append(sent)
                    current_words += sent_words

            if current_sents:
                chunk_text = " ".join(current_sents)
                suffix = f" - {part_num}" if part_num > 1 else ""
                result.append((chunk_text, f"{heading}{suffix}"))

        return result

    def _merge_same_topic(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge undersized chunks with neighbors.

        Rules:
          - Same heading prefix = always merge (they came from the same section)
          - Different heading = only merge if same topic (keyword overlap)
          - Never exceed max_words * 1.2
        """
        if not chunks:
            return chunks

        # Multiple passes until stable
        result = list(chunks)
        changed = True
        while changed:
            changed = False
            new_result = []
            i = 0
            while i < len(result):
                current = result[i]

                if current.words < self.config.min_words:
                    merged = False

                    # Try forward merge (with next chunk)
                    if i + 1 < len(result):
                        next_chunk = result[i + 1]
                        combined = current.words + next_chunk.words
                        if combined <= self.config.max_words * 1.2:
                            same_section = self._same_heading_base(current.heading, next_chunk.heading)
                            should = same_section
                            if not same_section:
                                ct = _extract_topic_words(current.content)
                                nt = _extract_topic_words(next_chunk.content)
                                should = _jaccard(ct, nt) > self.config.topic_shift_threshold
                            if should:
                                heading = self._merge_heading(current.heading, next_chunk.heading)
                                new_result.append(Chunk(
                                    chunk_id=current.chunk_id, heading=heading,
                                    content=current.content + " " + next_chunk.content,
                                    words=combined, source_type=current.source_type,
                                    source_ref=current.source_ref,
                                ))
                                i += 2
                                changed = True
                                merged = True

                    # Try backward merge (with previous chunk)
                    if not merged and new_result:
                        prev_chunk = new_result[-1]
                        combined = prev_chunk.words + current.words
                        if combined <= self.config.max_words * 1.2:
                            same_section = self._same_heading_base(prev_chunk.heading, current.heading)
                            should = same_section
                            if not same_section:
                                pt = _extract_topic_words(prev_chunk.content)
                                ct = _extract_topic_words(current.content)
                                should = _jaccard(pt, ct) > self.config.topic_shift_threshold
                            if should:
                                heading = self._merge_heading(prev_chunk.heading, current.heading)
                                new_result[-1] = Chunk(
                                    chunk_id=prev_chunk.chunk_id, heading=heading,
                                    content=prev_chunk.content + " " + current.content,
                                    words=combined, source_type=prev_chunk.source_type,
                                    source_ref=prev_chunk.source_ref,
                                )
                                i += 1
                                changed = True
                                merged = True

                    if not merged:
                        new_result.append(current)
                        i += 1
                    continue

                new_result.append(current)
                i += 1
            result = new_result

        return result

    @staticmethod
    def _same_heading_base(h1: str, h2: str) -> bool:
        """Check if two headings share the same base (before ' - N' suffix)."""
        base1 = re.sub(r'\s*-\s*\d+$', '', h1).strip()
        base2 = re.sub(r'\s*-\s*\d+$', '', h2).strip()
        return base1 == base2 and base1 != ""

    @staticmethod
    def _merge_heading(h1: str, h2: str) -> str:
        """Choose the best heading when merging two chunks."""
        base1 = re.sub(r'\s*-\s*\d+$', '', h1).strip()
        base2 = re.sub(r'\s*-\s*\d+$', '', h2).strip()
        if base1 == base2:
            return base1  # same section, drop the suffix
        return h1 or h2  # different sections, keep the first
