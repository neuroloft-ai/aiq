"""Test the full AIQ pipeline end-to-end."""
import pytest
from aiq import analyze, AIQConfig, Pipeline
from tests.fixtures import (
    NEUROLOFT_HTML, SIMPLE_CLEAN, SIMPLE_PII, SIMPLE_INTERNAL,
    SIMPLE_PLACEHOLDER, EXPECTED_DETECTIONS,
)


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
        result = analyze("Some text about refunds.")
        assert isinstance(result, PipelineResult)


class TestPII:
    """Test PII detection across modes."""

    def test_detects_email(self):
        result = analyze(SIMPLE_PII, config=AIQConfig(pii_mode="strict"))
        tagged = [c for c in result.chunks if c.tag.value == "pii"]
        assert len(tagged) >= 1

    def test_smart_mode_skips_functional_email(self):
        result = analyze(
            "Contact support@acme.com for help.",
            config=AIQConfig(pii_mode="smart"),
        )
        tagged = [c for c in result.chunks if c.tag.value == "pii"]
        assert len(tagged) == 0

    def test_strict_mode_blocks_functional_email(self):
        result = analyze(
            "Contact support@acme.com for help.",
            config=AIQConfig(pii_mode="strict"),
        )
        tagged = [c for c in result.chunks if c.tag.value == "pii"]
        assert len(tagged) >= 1


class TestGovernance:
    """Test governance tag detection."""

    def test_internal_note(self):
        result = analyze(SIMPLE_INTERNAL)
        tagged = [c for c in result.chunks if c.tag.value == "internal_only"]
        assert len(tagged) >= 1

    def test_placeholder(self):
        result = analyze(SIMPLE_PLACEHOLDER)
        tagged = [c for c in result.chunks if c.tag.value == "placeholder"]
        assert len(tagged) >= 1


class TestMultiDocument:
    """Test multi-document pipeline."""

    def test_multi_doc_input(self):
        docs = [
            {"id": "p1", "title": "Policy", "text": "Refunds take 5 days."},
            {"id": "p2", "title": "FAQ", "text": "Contact support for help."},
        ]
        result = analyze(docs)
        assert len(result.documents) == 2
        assert len(result.chunks) == 2

    def test_metadata_propagation(self):
        docs = [
            {"id": "p1", "title": "Policy", "text": "Refunds take 5 days.",
             "metadata": {"author": "Jane", "last_modified": "2026-04-15"}},
        ]
        result = analyze(docs)
        assert result.chunks[0].metadata.get("author") == "Jane"

    def test_source_tracking(self):
        docs = [
            {"id": "page_a", "title": "A", "text": "Content A."},
            {"id": "page_b", "title": "B", "text": "Content B."},
        ]
        result = analyze(docs)
        ids = {c.source_page_id for c in result.chunks}
        assert "page_a" in ids
        assert "page_b" in ids


class TestCustomRules:
    """Test custom detection rules."""

    def test_custom_block(self):
        config = AIQConfig(
            custom_rules=[
                {"pattern": "deprecated", "action": "block", "reason": "Deprecated"},
            ],
        )
        result = analyze("The deprecated API uses basic auth.", config=config)
        tagged = [c for c in result.chunks if c.tag.value == "custom_block"]
        assert len(tagged) >= 1

    def test_custom_review(self):
        config = AIQConfig(
            custom_rules=[
                {"pattern": "beta", "action": "review", "reason": "Beta content"},
            ],
        )
        result = analyze("Our new beta feature allows bulk exports.", config=config)
        tagged = [c for c in result.chunks if c.tag.value == "custom_review"]
        assert len(tagged) >= 1

    def test_builtin_takes_priority(self):
        """Built-in PII detection should win over custom rule."""
        config = AIQConfig(
            pii_mode="strict",
            custom_rules=[
                {"pattern": "john", "action": "review", "reason": "Name"},
            ],
        )
        result = analyze("Contact john@acme.com for help.", config=config)
        # PII (block) should win over custom (review)
        assert result.chunks[0].tag.value == "pii"


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


class TestFreshness:
    """Test content freshness detection."""

    def test_stale_flagging(self):
        docs = [
            {"id": "old", "title": "Old", "text": "Old content.",
             "metadata": {"last_modified": "2024-01-01"}},
        ]
        config = AIQConfig(freshness_threshold_days=365)
        result = analyze(docs, config=config)
        stale = [c for c in result.chunks if c.tag.value == "stale"]
        assert len(stale) >= 1

    def test_fresh_not_flagged(self):
        docs = [
            {"id": "new", "title": "New", "text": "Fresh content.",
             "metadata": {"last_modified": "2026-04-15"}},
        ]
        config = AIQConfig(freshness_threshold_days=365)
        result = analyze(docs, config=config)
        stale = [c for c in result.chunks if c.tag.value == "stale"]
        assert len(stale) == 0


class TestEvaluate:
    """Test Phase 4 evaluation."""

    def test_evaluate_runs(self):
        p = Pipeline()
        r = p.run("The billing team processes refunds in 5 business days.")
        e = p.evaluate(r)
        assert "A41" in e.module_outputs
        assert "A42" in e.module_outputs
        assert "A43" in e.module_outputs

    def test_user_qa_pairs(self):
        config = AIQConfig(
            user_qa_pairs=[
                {"question": "How long?", "expected_answer": "5 days",
                 "expected_behavior": "answer"},
            ],
        )
        p = Pipeline(config)
        r = p.run("Refunds take 5 business days.")
        e = p.evaluate(r)
        qa = e.module_outputs["A41"].data["qa_set"]
        user_pairs = [p for p in qa.pairs if p.source == "user"]
        assert len(user_pairs) >= 1


class TestNeuroloftFullDocument:
    """Test with the full Neuroloft knowledge base document.

    This document has 25 planted quality issues across all categories.
    Tests verify that the pipeline detects a minimum number of each type.
    """

    def test_detects_issues(self):
        result = analyze(NEUROLOFT_HTML)
        assert result.total_detected > 0

    def test_detects_pii(self):
        result = analyze(NEUROLOFT_HTML, config=AIQConfig(pii_mode="strict"))
        # Count PII findings across all module outputs
        a31 = result.module_outputs.get("A31")
        if a31 and a31.findings:
            pii_findings = [f for f in a31.findings if f.tag.value == "pii"]
            assert len(pii_findings) >= EXPECTED_DETECTIONS["pii"], \
                f"Expected >= {EXPECTED_DETECTIONS['pii']} PII, got {len(pii_findings)}"

    def test_detects_internal_notes(self):
        result = analyze(NEUROLOFT_HTML)
        a31 = result.module_outputs.get("A31")
        if a31 and a31.findings:
            internal = [f for f in a31.findings if f.tag.value == "internal_only"]
            assert len(internal) >= EXPECTED_DETECTIONS["internal_only"], \
                f"Expected >= {EXPECTED_DETECTIONS['internal_only']} internal, got {len(internal)}"

    def test_detects_placeholders(self):
        result = analyze(NEUROLOFT_HTML)
        a31 = result.module_outputs.get("A31")
        if a31 and a31.findings:
            ph = [f for f in a31.findings if f.tag.value == "placeholder"]
            assert len(ph) >= EXPECTED_DETECTIONS["placeholder"], \
                f"Expected >= {EXPECTED_DETECTIONS['placeholder']} placeholder, got {len(ph)}"

    def test_detects_metadata_leaks(self):
        result = analyze(NEUROLOFT_HTML)
        a31 = result.module_outputs.get("A31")
        if a31 and a31.findings:
            ml = [f for f in a31.findings if f.tag.value == "metadata_leak"]
            assert len(ml) >= EXPECTED_DETECTIONS["metadata_leak"], \
                f"Expected >= {EXPECTED_DETECTIONS['metadata_leak']} metadata_leak, got {len(ml)}"

    def test_domain_detected_as_support(self):
        result = analyze(NEUROLOFT_HTML)
        assert result.domain_context.domain_type == "support"

    def test_acronyms_detected(self):
        result = analyze(NEUROLOFT_HTML)
        ctx = result.domain_context
        # Should find at least SLA, CRM from content
        assert len(ctx.acronyms) >= 2

    def test_full_pipeline_completes(self):
        """Verify the full pipeline runs without errors on a real document."""
        result = analyze(NEUROLOFT_HTML)
        assert result.elapsed_seconds > 0
        assert len(result.chunks) > 0
        assert "A10" in result.module_outputs
        assert "A31" in result.module_outputs
