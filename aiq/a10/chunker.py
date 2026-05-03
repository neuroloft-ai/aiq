"""A10 — Raw Chunker

Separator-based chunking with size limits. Produces the "before" baseline
that AIQ measures improvement against.

What it does:
    Splits raw document text (plain text or HTML) into retrieval-sized chunks
    at natural boundaries (paragraphs, blank lines, HTML block elements).

Algorithm:
    1. Split text into blocks at paragraph/HTML boundaries
    2. Accumulate blocks into chunks respecting max_words limit
    3. Split oversized chunks at sentence boundaries
    4. Merge undersized chunks with neighbors
    5. Assign sequential IDs: raw_1, raw_2, ...

What it does NOT do:
    No heading detection, no topic awareness, no normalization, no content
    analysis. This is intentionally naive — it represents what a basic
    retrieval system gets without AIQ processing.

Config:
    min_words: minimum chunk size in words (default: 50)
    max_words: maximum chunk size in words (default: 200)
    strip_html: whether to strip HTML tags from content (default: True)

Config exposed to AIQConfig:
    chunk_min_words  -> A10Config.min_words   (default: 50)
    chunk_max_words  -> A10Config.max_words   (default: 200)
    strip_html       -> A10Config.strip_html  (default: True)

Input:  raw text (str), optional source_ref (str)
Output: ModuleOutput with .chunks = list[Chunk]

LLM required: No
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from aiq.core.types import Chunk, ModuleOutput


@dataclass
class A10Config:
    """Configuration for raw chunking."""
    min_words: int = 50
    max_words: int = 200
    # Separators in priority order
    # Blank lines and HTML block boundaries are primary separators
    strip_html: bool = True  # strip HTML tags from content


# Sentence boundary: period/question/exclamation followed by space and uppercase
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# HTML tag stripper
_TAG_RE = re.compile(r'<[^>]+>')

# Whitespace normalizer
_WS_RE = re.compile(r'\s+')


def _clean_block(text: str) -> str:
    """Strip HTML tags from a single block and normalize whitespace."""
    # Block-level closing tags get a period + space to preserve sentence boundaries
    text = re.sub(r'</(?:p|li|tr|div|h[1-6])>', '. ', text, flags=re.IGNORECASE)
    # Remaining tags get a space
    text = _TAG_RE.sub(' ', text)
    # Clean up artifacts
    text = re.sub(r':\s*\.', ':', text)
    text = re.sub(r'\.(\s*\.)+', '.', text)
    text = _WS_RE.sub(' ', text)
    return text.strip()


def _split_into_blocks(text: str) -> list[str]:
    """Split text into blocks at paragraph/element boundaries.

    Uses a simple approach: replace block-level HTML tags with a
    unique delimiter, then split on that delimiter. Each block
    is cleaned individually to preserve paragraph structure.
    """
    _DELIM = "\n\x00SPLIT\x00\n"

    # Insert delimiter before opening block tags
    out = re.sub(
        r'<(?:p|div|h[1-6]|ul|ol|table|thead|tbody)[\s>]',
        lambda m: _DELIM + m.group(),
        text,
        flags=re.IGNORECASE,
    )
    # Insert delimiter after closing block tags
    out = re.sub(
        r'</(?:p|div|h[1-6]|ul|ol|table|thead|tbody)>',
        lambda m: m.group() + _DELIM,
        out,
        flags=re.IGNORECASE,
    )
    # Also split on blank lines
    out = re.sub(r'\n\s*\n', _DELIM, out)

    raw_blocks = out.split(_DELIM)

    # Clean each block and filter empties
    result = []
    for block in raw_blocks:
        cleaned = _clean_block(block)
        if cleaned and len(cleaned.split()) > 3:
            result.append(cleaned)
    return result


def _word_count(text: str) -> int:
    return len(text.split())


def _find_sentence_break(text: str, max_words: int) -> int:
    """Find the best position to break text at a sentence boundary within max_words.

    Returns the character index to break at, or -1 if no good break found.
    """
    words = text.split()
    if len(words) <= max_words:
        return len(text)

    # Build text up to max_words
    target_text = ' '.join(words[:max_words])
    # Find the last sentence boundary within this range
    last_end = -1
    for m in _SENT_RE.finditer(target_text):
        last_end = m.start()

    if last_end > len(target_text) * 0.3:
        return last_end

    # No good sentence boundary — fall back to word boundary at max_words
    return len(target_text)


class RawChunker:
    """A10 — Naive separator-based chunker for baseline comparison."""

    def __init__(self, config: Optional[A10Config] = None):
        self.config = config or A10Config()

    def run(self, text: str, source_ref: str = "") -> ModuleOutput:
        """Chunk raw text into retrieval-sized pieces.

        Args:
            text: raw document text (may contain HTML)
            source_ref: source reference (e.g., page title, file name)

        Returns:
            ModuleOutput with raw chunks in .chunks
        """
        t0 = time.perf_counter()

        if not text.strip():
            return ModuleOutput(
                module_id="A10",
                module_name="Raw Chunking",
                elapsed_seconds=time.perf_counter() - t0,
            )

        # Split into blocks (splits on HTML structure, then cleans each block)
        blocks = _split_into_blocks(text)

        # If no blocks found (continuous text), treat whole text as one block
        if not blocks:
            blocks = [_clean_block(text)]

        # Build chunks by accumulating blocks within size limits
        chunks = []
        current_text = ""
        current_words = 0

        for block in blocks:
            block_words = _word_count(block)

            # If adding this block exceeds max, finalize current chunk
            if current_words > 0 and current_words + block_words > self.config.max_words:
                chunks.append(current_text.strip())
                current_text = block
                current_words = block_words
            else:
                separator = " " if current_text else ""
                current_text += separator + block
                current_words += block_words

        # Don't forget the last chunk
        if current_text.strip():
            chunks.append(current_text.strip())

        # Post-process: split oversized chunks at sentence boundaries
        final_chunks = []
        for chunk_text in chunks:
            wc = _word_count(chunk_text)
            if wc > self.config.max_words:
                # Split at sentence boundaries
                parts = self._split_oversized(chunk_text)
                final_chunks.extend(parts)
            else:
                final_chunks.append(chunk_text)

        # Post-process: merge undersized chunks with neighbors
        merged_chunks = self._merge_undersized(final_chunks)

        # Build Chunk objects
        output_chunks = []
        for i, text in enumerate(merged_chunks):
            wc = _word_count(text)
            if wc < 3:  # skip near-empty chunks
                continue
            output_chunks.append(Chunk(
                chunk_id=f"raw_{i + 1}",
                heading="",  # raw chunking has no heading awareness
                content=text,
                words=wc,
                source_type="text",
                source_ref=source_ref,
            ))

        total_words = sum(c.words for c in output_chunks)
        return ModuleOutput(
            module_id="A10",
            module_name="Raw Chunking",
            detected=0,    # A10 doesn't detect issues
            resolved=0,
            remaining=0,
            words_in=total_words,
            words_out=total_words,
            chunks=output_chunks,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _split_oversized(self, text: str) -> list[str]:
        """Split an oversized chunk at sentence boundaries."""
        result = []
        remaining = text

        while _word_count(remaining) > self.config.max_words:
            break_pos = _find_sentence_break(remaining, self.config.max_words)
            if break_pos <= 0 or break_pos >= len(remaining):
                # Can't find a good break, keep as-is
                break
            part = remaining[:break_pos].strip()
            if part:
                result.append(part)
            remaining = remaining[break_pos:].strip()

        if remaining.strip():
            result.append(remaining.strip())

        return result if result else [text]

    def _merge_undersized(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are below min_words with their neighbor."""
        if not chunks:
            return chunks

        result = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            # If undersized and there's a next chunk, merge forward
            while (_word_count(current) < self.config.min_words
                   and i + 1 < len(chunks)
                   and _word_count(current) + _word_count(chunks[i + 1]) <= self.config.max_words * 1.2):
                i += 1
                current = current + " " + chunks[i]
            result.append(current)
            i += 1

        return result
