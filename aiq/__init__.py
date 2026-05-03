"""AIQ — Improving Knowledge Base Quality to Optimize Retrieval in RAG Systems.

A modular data-centric pipeline that detects and resolves content quality
issues in knowledge bases before they enter the retrieval layer.

Quick start:
    from aiq import analyze, AIQConfig

    # Simple — rule-based, no LLM needed
    result = analyze("your document text or HTML")
    print(f"Issues found: {result.total_detected}")
    print(f"Issues resolved: {result.total_resolved}")

    # With LLM for better quality
    result = analyze("your text", config=AIQConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
    ))

    # From files
    from aiq.loader import load_file
    result = analyze(load_file("policy.docx"))

Pipeline phases:
    Phase 1 (A10-A14): Intake & Structure
    Phase 2 (A22):     Metadata Enrichment
    Phase 3 (A30-A32): Detection & Fix (clarity, governance, consistency)
    Phase 4 (A41-A43): Evaluation (optional, via pipeline.evaluate())
"""

from aiq.pipeline import Pipeline, AIQConfig, PipelineConfig, PipelineResult


def analyze(input_data, config=None):
    """Run the AIQ pipeline on documents.

    Args:
        input_data: document text (str), file dict, or list of file dicts
        config: AIQConfig (optional — defaults work without LLM)

    Returns:
        PipelineResult with chunks, findings, and metrics
    """
    return Pipeline(config).run(input_data)


__all__ = ["analyze", "Pipeline", "AIQConfig", "PipelineConfig", "PipelineResult"]
