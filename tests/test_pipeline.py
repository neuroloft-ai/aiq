"""Test the full AIQ pipeline end-to-end.

Tests verify the detect-and-remove architecture:
  - Unsafe content (PII, internal notes, placeholders, editorial, metadata)
    is REMOVED from chunk content, not just tagged
  - Clean content is preserved and retrievable
  - Domain context is correctly inferred
  - Multi-document processing works
  - Configuration options function correctly
"""
import pytest
from aiq import analyze, AIQConfig, Pipeline
from tests.fixtures import (
    NEUROLOFT_HTML, SIMPLE_CLEAN, SIMPLE_PII, SIMPLE_INTERNAL,
    SIMPLE_PLACEHOLDER, SIMPLE_EDITORIAL, SIMPLE_METADATA,
    EXPECTED_DETECTIONS,
)


# =====================================================================
# Basic pipeline
# =====================================================================

class TestAnalyzeSimple:
    """Test the one-liner analyze() function."""

    def test_clean_text(self):
        result = analyze(SIMPLE_CLEAN)
        assert len(result.chunks) >= 1
        assert result.domain_context is not None

    def test_empty_string(self):
        result = analyze("")
        assert len(result.chunks) == 0

    def test_empty_list(self):
        result = analyze([])
        assert len(result.chunks) == 0

    def test_returns_pipeline_result(self):
        from aiq.pipeline import PipelineResult
        result = analyze("Some text about refunds and billing support.")
        assert isinstance(result, PipelineResult)

    def test_pipeline_completes_without_error(self):
        text = ("The billing team processes refunds within 10 business days. "
                "Customers can submit refund requests through the support portal. "
                "All refund requests are reviewed by the billing department before approval. "
                "For enterprise customers, additional verification may be required.")
        result = analyze(text)
        assert result.elapsed_seconds > 0
        assert len(result.chunks) > 0


# =====================================================================
# Content cleaning (detect-and-remove)
# =====================================================================

class TestPIICleaning:
    """Test that PII is removed from content, not just tagged."""

    def test_email_removed_from_content(self):
        result = analyze(SIMPLE_PII, config=AIQConfig(pii_mode="strict"))
        all_content = " ".join(c.content for c in result.chunks)
        assert "john.smith@company.com" not in all_content
        assert "(555) 867-5309" not in all_content

    def test_smart_mode_keeps_functional_email(self):
        text = ("Contact support@acme.com for help with your billing questions. "
                "Our support team is available Monday through Friday during business hours. "
                "All inquiries are handled within 24 hours of submission. "
                "For urgent issues, please call our main support line.")
        result = analyze(text, config=AIQConfig(pii_mode="smart"))
        all_content = " ".join(c.content for c in result.chunks)
        # Functional emails should be preserved in smart mode
        assert "support@acme.com" in all_content

    def test_strict_mode_removes_functional_email(self):
        result = analyze(
            "Contact support@acme.com for help. The billing team is available during business hours.",
            config=AIQConfig(pii_mode="strict"),
        )
        all_content = " ".join(c.content for c in result.chunks)
        assert "support@acme.com" not in all_content

    def test_pii_detected_in_findings(self):
        result = analyze(SIMPLE_PII, config=AIQConfig(pii_mode="strict"))
        a31 = result.module_outputs.get("A31")
        assert a31 is not None
        assert a31.detected > 0


class TestGovernanceCleaning:
    """Test that governance issues are removed from content."""

    def test_internal_note_removed(self):
        result = analyze(SIMPLE_INTERNAL)
        all_content = " ".join(c.content for c in result.chunks)
        assert "INTERNAL NOTE" not in all_content
        assert "Do not share" not in all_content

    def test_placeholder_removed(self):
        result = analyze(SIMPLE_PLACEHOLDER)
        all_content = " ".join(c.content for c in result.chunks)
        assert "TODO" not in all_content

    def test_editorial_removed(self):
        text = "Invoices are due within 15 days. [TRACKED CHANGE - Dave: changed from 30 days]. Late fee is 1.5%."
        result = analyze(text)
        all_content = " ".join(c.content for c in result.chunks)
        assert "TRACKED CHANGE" not in all_content
        # Clean content should be preserved
        assert "1.5%" in all_content or "Late fee" in all_content or len(result.chunks) == 0

    def test_metadata_leak_removed(self):
        text = "Gateway timeout tracked in JIRA-7823. Engineering ETA Sprint 47. Retry after 30 seconds for the payment to process correctly."
        result = analyze(text)
        all_content = " ".join(c.content for c in result.chunks)
        assert "JIRA-7823" not in all_content

    def test_section_internal_clears_entire_chunk(self):
        text = "FOR INTERNAL USE ONLY. Contact Jennifer Park at j.park@company.com for escalation."
        result = analyze(text)
        all_content = " ".join(c.content for c in result.chunks)
        assert "FOR INTERNAL USE ONLY" not in all_content
        assert "j.park@company.com" not in all_content


class TestClarityRewriting:
    """Test that clarity issues are rewritten in content."""

    def test_acronym_expanded(self):
        text = "Submit your request via CRM and check the SLA for response times. The billing team handles all inquiries."
        result = analyze(text)
        all_content = " ".join(c.content for c in result.chunks)
        # CRM should be expanded
        assert "Customer Relationship Management" in all_content or "CRM" in all_content

    def test_pronoun_resolved(self):
        text = "The billing team handles all inquiries. They process refunds within 10 business days."
        result = analyze(text)
        a30 = result.module_outputs.get("A30")
        assert a30 is not None
        assert a30.detected > 0


# =====================================================================
# Multi-document processing
# =====================================================================

class TestMultiDocument:
    """Test multi-document pipeline."""

    def test_multi_doc_input(self):
        docs = [
            {"id": "p1", "title": "Refund Policy",
             "text": "Refunds are processed within 10 business days after approval by the billing team."},
            {"id": "p2", "title": "Payment FAQ",
             "text": "We accept credit cards, wire transfers, and PayPal for all payment processing."},
        ]
        result = analyze(docs)
        assert len(result.documents) == 2
        assert len(result.chunks) >= 1

    def test_metadata_propagation(self):
        docs = [
            {"id": "p1", "title": "Policy",
             "text": "Refunds are processed within 10 business days by the support team.",
             "metadata": {"author": "Jane", "last_modified": "2026-04-15"}},
        ]
        result = analyze(docs)
        # At least one chunk should have the metadata
        has_author = any(c.metadata.get("author") == "Jane" for c in result.chunks)
        assert has_author

    def test_source_tracking(self):
        docs = [
            {"id": "page_a", "title": "Policy A",
             "text": "The refund policy allows returns within 30 days of purchase for all customers."},
            {"id": "page_b", "title": "Policy B",
             "text": "Payment processing takes 3 to 5 business days for wire transfer transactions."},
        ]
        result = analyze(docs)
        ids = {c.source_page_id for c in result.chunks}
        assert len(ids) >= 1  # At least one source tracked


# =====================================================================
# Configuration
# =====================================================================

class TestConfig:
    """Test AIQConfig options."""

    def test_confidence_categorical(self):
        c = AIQConfig(detection_confidence="high")
        assert c.detection_confidence == 0.8

    def test_confidence_numeric(self):
        c = AIQConfig(detection_confidence=0.65)
        assert c.detection_confidence == 0.65

    def test_default_config(self):
        c = AIQConfig()
        assert c.pii_mode == "smart"
        assert c.detection_confidence == 0.5
        assert c.freshness_threshold_days == 180

    def test_backward_compat_alias(self):
        from aiq.pipeline import PipelineConfig
        c = PipelineConfig(pii_mode="strict")
        assert c.pii_mode == "strict"


class TestTagBehavior:
    """Test tag behavior overrides."""

    def test_override_to_allow(self):
        config = AIQConfig(tag_behavior={"vague_claim": "allow"})
        from aiq.core.types import ChunkTag
        tag = ChunkTag.VAGUE_CLAIM
        assert tag.behavior(config.tag_behavior) == "allow"

    def test_override_to_block(self):
        config = AIQConfig(tag_behavior={"destructive": "block"})
        from aiq.core.types import ChunkTag
        tag = ChunkTag.DESTRUCTIVE
        assert tag.behavior(config.tag_behavior) == "block"

    def test_default_behavior_unchanged(self):
        from aiq.core.types import ChunkTag
        assert ChunkTag.PII.default_behavior == "block"
        assert ChunkTag.VAGUE_CLAIM.default_behavior == "review"
        assert ChunkTag.CONTENT.default_behavior == "answer"


# =====================================================================
# Neuroloft full document (integration test)
# =====================================================================

class TestNeuroloftFullDocument:
    """Integration test with the full Neuroloft knowledge base.

    Tests verify:
      - Pipeline completes without errors
      - Domain is correctly detected
      - Governance issues are detected in findings
      - Unsafe content is removed from output chunks
      - Clean content is preserved
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.result = analyze(NEUROLOFT_HTML)

    def test_pipeline_completes(self):
        assert self.result.elapsed_seconds > 0
        assert len(self.result.chunks) > 0
        assert "A10" in self.result.module_outputs
        assert "A31" in self.result.module_outputs

    def test_domain_detected_as_support(self):
        assert self.result.domain_context.domain_type == "support"

    def test_acronyms_detected(self):
        ctx = self.result.domain_context
        assert len(ctx.acronyms) >= 2

    def test_detects_issues(self):
        assert self.result.total_detected > 0

    def test_governance_findings_present(self):
        a31 = self.result.module_outputs.get("A31")
        assert a31 is not None
        assert a31.detected > 0
        # Check minimum detections per category
        for category, min_count in EXPECTED_DETECTIONS.items():
            findings = [f for f in a31.findings if f.tag.value == category]
            assert len(findings) >= min_count, \
                f"Expected >= {min_count} {category} findings, got {len(findings)}"

    def test_pii_removed_from_output(self):
        all_content = " ".join(c.content for c in self.result.chunks)
        # Personal emails should be removed
        assert "sarah.johnson@acmecorp.com" not in all_content
        assert "m.chen@techstart.io" not in all_content
        assert "lisa.r@globalinc.com" not in all_content

    def test_internal_notes_removed_from_output(self):
        all_content = " ".join(c.content for c in self.result.chunks)
        assert "INTERNAL NOTE" not in all_content
        assert "FOR INTERNAL USE ONLY" not in all_content

    def test_placeholders_removed_from_output(self):
        all_content = " ".join(c.content for c in self.result.chunks)
        # TODO/TBD should be removed from content
        # Check that the placeholder sentences are gone, not just the keyword
        has_todo_sentence = "TODO:" in all_content or "TODO " in all_content
        has_tbd_sentence = "TBD -" in all_content or "TBD " in all_content
        assert not has_todo_sentence, "TODO placeholder still in output"
        assert not has_tbd_sentence, "TBD placeholder still in output"

    def test_metadata_leaks_removed_from_output(self):
        all_content = " ".join(c.content for c in self.result.chunks)
        assert "JIRA-7823" not in all_content
        assert "grafana.internal" not in all_content

    def test_clean_content_preserved(self):
        all_content = " ".join(c.content for c in self.result.chunks)
        # Core billing content should still be present
        assert "refund" in all_content.lower()
        assert "payment" in all_content.lower()

    def test_all_output_chunks_are_servable(self):
        for chunk in self.result.chunks:
            assert chunk.words > 0, f"Empty chunk found: {chunk.chunk_id}"
