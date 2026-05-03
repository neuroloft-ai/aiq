"""A10 — Raw Chunking.

Separator-based chunking of raw document content. No normalization,
no structure awareness. Just break text into chunks at natural
boundaries (paragraphs, blank lines) with configurable size limits.

Output = raw chunks used as "before" baseline for retrieval testing.
The ONLY purpose is to capture what the data looks like before any
AIQ processing, so we can measure the improvement.
"""

from .chunker import RawChunker, A10Config

__all__ = ["RawChunker", "A10Config"]
