"""AIQ Core Types — the foundation for all modules.

Three foundational pieces:
  1. Chunk — the retrieval unit with classification tag
  2. ModuleOutput — standard output (Detected, Resolved, Remaining)
  3. DomainContext — shared vocabulary consumed by all modules

Design principles:
  - No module has its own hardcoded vocabulary
  - All domain knowledge flows through DomainContext
  - Every module produces ModuleOutput with the same three counts
  - Chunks are tagged, not edited/removed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =====================================================================
# Chunk Tags — classification for retrieval behavior
# =====================================================================

class ChunkTag(str, Enum):
    """Classification tag that drives retrieval behavior.

    Two categories:
      Auto-block: blocked from retrieval by default (user can override)
      User-review: flagged for user decision
    """
    # Normal content — retrievable
    CONTENT = "content"

    # Auto-block (default: blocked from retrieval)
    INTERNAL_ONLY = "internal_only"
    PLACEHOLDER = "placeholder"
    PII = "pii"
    EDITORIAL = "editorial"
    METADATA_LEAK = "metadata_leak"

    # User-review (flagged, user decides)
    VAGUE_CLAIM = "vague_claim"
    DESTRUCTIVE = "destructive"
    BROKEN_REFERENCE = "broken_reference"
    ESCALATION = "escalation"

    # User-driven (optional, based on metadata)
    STALE = "stale"

    # Custom rule match (from PipelineConfig.custom_rules)
    CUSTOM_BLOCK = "custom_block"
    CUSTOM_REVIEW = "custom_review"

    @property
    def default_behavior(self) -> str:
        """Default retrieval behavior for this tag.

        Can be overridden per-tag via PipelineConfig.tag_behavior.
        Use behavior(overrides) for runtime override support.
        """
        auto_block = {
            self.INTERNAL_ONLY, self.PLACEHOLDER, self.PII,
            self.EDITORIAL, self.METADATA_LEAK, self.CUSTOM_BLOCK,
        }
        if self == self.CONTENT:
            return "answer"
        if self in auto_block:
            return "block"
        return "review"

    def behavior(self, overrides: dict = None) -> str:
        """Retrieval behavior with optional overrides.

        Args:
            overrides: {tag_value: "block"|"review"|"allow"} from PipelineConfig.tag_behavior

        Returns:
            "answer" (serve), "block" (exclude), "review" (serve with caveat), "allow" (serve, ignore tag)
        """
        if overrides and self.value in overrides:
            return overrides[self.value]
        return self.default_behavior

    def caveat_message(self) -> str:
        """Warning message to surface when a review-tagged chunk is retrieved.

        Downstream RAG systems should include this caveat when serving the
        chunk as an answer, so users know the limitation.

        Returns empty string for block tags (content not served) and content tags.
        """
        caveats = {
            self.VAGUE_CLAIM: (
                "This answer contains vague language that may not give specific guidance. "
                "Contact support for precise details."
            ),
            self.BROKEN_REFERENCE: (
                "This answer references a resource that is not available in the knowledge base. "
                "The referenced content may be missing or outdated."
            ),
            self.ESCALATION: (
                "This answer describes an escalation without specific contact details. "
                "You may need to check with your team for the correct escalation path."
            ),
            self.DESTRUCTIVE: (
                "This answer describes a destructive action. "
                "Please verify the context and prerequisites before proceeding."
            ),
            self.STALE: (
                "This content may be outdated based on when the source was last updated. "
                "Verify with a current source."
            ),
        }
        return caveats.get(self, "")

    def requires_caveat(self) -> bool:
        """Whether a chunk with this tag should carry a caveat message when retrieved."""
        return bool(self.caveat_message())


# =====================================================================
# Chunk — the retrieval unit
# =====================================================================

@dataclass
class Chunk:
    """A retrieval unit — the fundamental object that flows through the pipeline.

    Core retrieval unit with classification support:
      - tag: ChunkTag classification for retrieval behavior
      - tag_reason: why this tag was assigned
      - tag_module: which module assigned it
      - source_ref: reference back to source (table, figure, page, section)
      - token_changes: list of token additions/removals with reasons
    """
    chunk_id: str
    heading: str
    content: str
    words: int
    source_type: str = "text"       # text / table / figure / procedure / code
    parent_chunk_id: Optional[str] = None

    # Classification
    tag: ChunkTag = ChunkTag.CONTENT
    tag_reason: str = ""            # why this tag was assigned
    tag_module: str = ""            # which module assigned it (e.g., "A31")

    # Source reference — traceability back to original
    source_ref: str = ""            # e.g., "Table 3, Row 2", "Figure 4", "Page: Billing"
    source_page_id: str = ""
    source_page_title: str = ""

    # Metadata (module-enriched)
    metadata: dict = field(default_factory=dict)

    # Token change tracking
    token_changes: list = field(default_factory=list)  # list[TokenChange]

    def token_count(self) -> int:
        """Approximate token count (words * 1.3 rough estimate)."""
        return int(self.words * 1.3)


# =====================================================================
# Token Change Tracking
# =====================================================================

@dataclass
class TokenChange:
    """One token addition or removal with reason.

    Added = gap filled (table extraction, heading generation, acronym expansion)
    Removed = waste eliminated (tagged as internal, placeholder, editorial)
    """
    change_type: str        # "added" or "removed"
    reason: str             # why: "table_extraction", "heading_generation",
                            #      "internal_tagged", "placeholder_tagged", etc.
    token_count: int        # how many tokens changed
    module: str             # which module made the change (e.g., "A12", "A31")
    detail: str = ""        # optional: what specifically changed


@dataclass
class TokenAccounting:
    """Aggregate token accounting for the full pipeline."""
    before_tokens: int = 0
    after_tokens: int = 0
    added: list = field(default_factory=list)    # list[TokenChange] where type="added"
    removed: list = field(default_factory=list)  # list[TokenChange] where type="removed"

    @property
    def total_added(self) -> int:
        return sum(tc.token_count for tc in self.added)

    @property
    def total_removed(self) -> int:
        return sum(tc.token_count for tc in self.removed)

    @property
    def net_change(self) -> int:
        return self.total_added - self.total_removed

    def summary_by_reason(self) -> dict:
        """Group token changes by reason."""
        result = {}
        for tc in self.added + self.removed:
            key = f"{tc.change_type}:{tc.reason}"
            result[key] = result.get(key, 0) + tc.token_count
        return result


# =====================================================================
# Module Output Standard — Detected, Resolved, Remaining
# =====================================================================

@dataclass
class ModuleOutput:
    """Standard output from every AIQ module.

    Three counts tell the full story:
      - Detected: what we found
      - Resolved: what we fixed
      - Remaining: what still needs attention

    No scores, no formulas. Just counts. The user sees exactly
    where they stand.
    """
    module_id: str          # e.g., "A11", "A31", "A42"
    module_name: str        # e.g., "Raw Chunking", "Classification"

    # The three counts
    detected: int = 0       # issues/items found
    resolved: int = 0       # issues/items handled
    remaining: int = 0      # issues still needing attention

    # Token tracking per module
    words_in: int = 0               # total words entering this module
    words_out: int = 0              # total words leaving this module

    # Details
    findings: list = field(default_factory=list)    # detailed finding objects
    chunks: list = field(default_factory=list)       # output chunks (if modified)
    token_changes: list = field(default_factory=list) # list[TokenChange]

    # Module-specific output data
    data: dict = field(default_factory=dict)  # e.g., {"normalized_html": "..."}

    # Execution info
    elapsed_seconds: float = 0.0
    error: str = ""

    @property
    def all_resolved(self) -> bool:
        return self.remaining == 0 and self.detected > 0


# =====================================================================
# Domain Context — shared vocabulary for all modules
# =====================================================================

@dataclass
class DomainContext:
    """Shared domain vocabulary consumed by ALL downstream modules.

    Produced by A21 (Domain Intelligence & Scope). No other module
    maintains its own hardcoded vocabulary. Everything reads from here.

    Can be:
      - Auto-inferred from content by A21
      - User-provided/overridden in Configure
      - Loaded from a domain profile (future: F21 knowledge base)
    """
    # Inferred domain type
    domain_type: str = ""   # e.g., "support", "medical", "legal", "engineering"
    confidence: float = 0.0 # how confident A11 is in the inference
    company_name: str = ""  # the organization that owns this KB

    # Topic vocabulary: {topic_name: set of synonyms}
    # e.g., {"refund": {"refund", "return", "reimburse", "money back"}}
    domain_anchors: dict = field(default_factory=dict)

    # Actor/team vocabulary: {role_word: normalized_name}
    # e.g., {"agent": "support", "billing dept": "billing"}
    actors: dict = field(default_factory=dict)

    # Known acronyms in this domain: {acronym: expansion}
    # e.g., {"SLA": "Service Level Agreement", "CRM": "Customer Relationship Management"}
    acronyms: dict = field(default_factory=dict)

    # Common acronyms that should NOT be flagged (cross-domain)
    common_acronyms: set = field(default_factory=lambda: {
        "US", "USA", "UK", "EU", "UN", "ID", "HR", "IT",
        "AM", "PM", "API", "URL", "PDF", "CSV", "JSON", "XML", "HTML",
        "HTTP", "HTTPS", "FAQ", "USD", "EUR", "GB", "MB", "KB", "TB",
        "CEO", "CTO", "CFO", "COO", "VP", "OK", "VS", "NA",
    })

    # Product/service names found in content
    product_names: list = field(default_factory=list)

    # Context normalization: maps variant terms to canonical form
    # e.g., {"invoice": "billing", "chargeback": "dispute"}
    context_normalization: dict = field(default_factory=dict)

    # Destructive action patterns for this domain
    # e.g., ["delete account", "purge data", "cancel subscription"]
    destructive_patterns: list = field(default_factory=list)

    # Business impact categories relevant to this domain
    # e.g., {"financial": "Wrong amounts cost money", "compliance": "Regulatory risk"}
    impact_categories: dict = field(default_factory=dict)

    # Scope qualifiers found across chunks
    # e.g., [{"type": "region", "values": ["US", "Asia", "Europe"]}]
    scope_qualifiers: list = field(default_factory=list)

    def is_known_acronym(self, acronym: str) -> bool:
        """Check if acronym is known (domain or common)."""
        return acronym in self.acronyms or acronym in self.common_acronyms

    def normalize_term(self, term: str) -> str:
        """Map a term to its canonical form, or return as-is."""
        return self.context_normalization.get(term.lower(), term.lower())
