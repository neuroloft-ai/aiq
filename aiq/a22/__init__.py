"""A22 — Metadata Enrichment.

Extracts dates, versions, and entities from chunk content and metadata.
Stores on chunk.metadata for downstream modules (A32 uses dates
for contradiction resolution).

Primary signal: Confluence page last_modified (if available).
Secondary: content dates extracted from text.
Optional: user-driven stale flagging (pages older than N months).
"""

from .enricher import MetadataEnricher, A22Config

__all__ = ["MetadataEnricher", "A22Config"]
