"""AIQ Pipeline — headless orchestrator for the full quality pipeline.

Supports single or multi-document input. Each document goes through
Phase 1 (A10-A14) independently, then all chunks are pooled for
Phase 3 (A30-A32) to enable cross-document consistency detection.
Phase 4 (A41-A43) is optional evaluation — run separately via evaluate().

Usage:
    from aiq.pipeline import Pipeline, AIQConfig

    # Simple — rule-based, no LLM needed
    result = Pipeline().run("your document text")

    # With LLM
    config = AIQConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
    )
    result = Pipeline(config).run("your document text")

    # Multi-document
    docs = [
        {"id": "page_1", "title": "Refund Policy", "text": "<html>..."},
        {"id": "page_2", "title": "Billing FAQ", "text": "<html>..."},
    ]
    result = Pipeline(config).run(docs)

    # From files
    from aiq.loader import load_file, load_directory
    result = Pipeline(config).run(load_file("policy.docx"))
    result = Pipeline(config).run(load_directory("kb_pages/"))

    # Evaluate (Phase 4 — optional, separate from processing)
    eval_result = Pipeline(config).evaluate(result)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Union

from aiq.core.types import Chunk, ChunkTag, DomainContext, ModuleOutput


# =====================================================================
# Pipeline Config — single config object for all modules
# =====================================================================

@dataclass
class AIQConfig:
    """Configuration for the full AIQ pipeline.

    All settings in one place. Internally maps to module-specific configs.
    The pipeline works without any LLM — all modules have rule-based fallbacks.
    """
    # ── LLM (optional — pipeline works without) ──
    # Option 1: specify provider + key (we create the client)
    llm_provider: str = ""                  # "openai" | "groq" | "anthropic" | "" (none)
    llm_api_key: str = ""                   # API key for the provider
    llm_model: str = ""                     # model name (uses provider default if empty)
    # Option 2: bring your own callable (overrides provider)
    llm_call: Optional[Callable] = None     # callable(prompt: str) -> str
    vision_call: Optional[Callable] = None  # callable(prompt: str, image_path: str) -> str

    # ── Phase 1: Intake ──
    chunk_min_words: int = 50
    chunk_max_words: int = 200
    strip_html: bool = True
    phase1_mode: str = "rule_only"        # "rule_only" | "rule_then_llm" | "llm_all"
    topic_shift_threshold: float = 0.1
    image_dir: str = ""

    # ── Phase 3: Detection ──
    pii_mode: str = "smart"               # "strict" | "smart" | "lenient"
    clarity_pronoun_mode: str = "rule_fix"
    clarity_acronym_mode: str = "rule_fix"
    consistency_llm_judge: bool = False

    # ── Domain overrides (applied after A11 auto-detection) ──
    domain_type: str = ""                  # override auto-detected domain
    company_name: str = ""                 # override auto-detected company
    extra_actors: dict = field(default_factory=dict)
    extra_acronyms: dict = field(default_factory=dict)
    extra_products: list = field(default_factory=list)
    extra_destructive: list = field(default_factory=list)

    # ── Detection confidence ──
    # Controls which findings get acted on (fixes, tags, flags).
    # Categorical: "high" (0.8), "medium" (0.5), "low" (0.3)
    # Numeric: any float 0.0-1.0 for fine control
    # Findings below threshold are still detected but not applied.
    detection_confidence: float = 0.5     # default: medium

    # ── Tag behavior overrides ──
    # Controls what happens when a tag is assigned to a chunk.
    # Default: PII/internal/placeholder/editorial/metadata_leak = "block"
    #          vague/destructive/broken_ref/escalation/stale = "review"
    # Override any tag: {"destructive": "block", "vague_claim": "allow"}
    # "block" = excluded from retrieval, "review" = served with caveat, "allow" = served normally
    tag_behavior: dict = field(default_factory=dict)

    # ── Custom detection rules ──
    # User-defined regex patterns to block or flag content beyond the 9 built-in detectors.
    # Each rule: {"pattern": "regex", "action": "block"|"review", "reason": "why"}
    # Examples:
    #   {"pattern": "beta feature", "action": "review", "reason": "Beta content needs review"}
    #   {"pattern": "competitor|competing", "action": "block", "reason": "Competitor mentions"}
    #   {"pattern": r"price:\s*\$\d+", "action": "review", "reason": "Pricing needs approval"}
    custom_rules: list = field(default_factory=list)

    # ── Metadata-based rules ──
    freshness_threshold_days: int = 180    # flag content older than this as stale (0 = disabled)
    priority_authors: list = field(default_factory=list)  # these authors' content wins in contradictions
    scope_filters: dict = field(default_factory=dict)     # e.g. {"region": "US", "tier": "Enterprise"}

    # ── Phase 4: Evaluation ──
    eval_total_questions: int = 0     # auto-generated question count (0 = auto-recommend)
    user_qa_pairs: list = field(default_factory=list)  # user-provided Q&A pairs
    # Each: {"question": str, "expected_answer": str, "expected_behavior": "answer"|"block"|"caveat"}
    # Merged with auto-generated pairs. User pairs are never auto-deleted.

    # ── Phases to run ──
    phases: list = field(default_factory=lambda: [1, 2, 3])  # [1, 2, 3]

    def __post_init__(self):
        """Resolve config shortcuts."""
        # Confidence: categorical -> numeric
        _CONFIDENCE_MAP = {"low": 0.3, "medium": 0.5, "high": 0.8}
        if isinstance(self.detection_confidence, str):
            self.detection_confidence = _CONFIDENCE_MAP.get(
                self.detection_confidence.lower(), 0.5
            )

        # LLM: create client from provider + key if no custom callable
        if not self.llm_call and self.llm_provider and self.llm_api_key:
            from aiq.llm import create_llm_client
            self.llm_call = create_llm_client(
                provider=self.llm_provider,
                api_key=self.llm_api_key,
                model=self.llm_model,
            )


# =====================================================================
# Pipeline Result
# =====================================================================

@dataclass
class PipelineResult:
    """Result from a full pipeline run."""
    # Final chunks (after all processing)
    chunks: list = field(default_factory=list)          # list[Chunk]
    raw_chunks: list = field(default_factory=list)       # list[Chunk] (A10 baseline)

    # Domain context
    domain_context: Optional[DomainContext] = None

    # Per-module outputs
    module_outputs: dict = field(default_factory=dict)   # {module_id: ModuleOutput}

    # Per-document info
    documents: list = field(default_factory=list)        # [{id, title, chunk_count}]

    # Timing
    elapsed_seconds: float = 0.0

    # Summary counts
    total_detected: int = 0
    total_resolved: int = 0
    total_remaining: int = 0


# =====================================================================
# Document normalization
# =====================================================================

@dataclass
class _Document:
    """Internal representation of a document for pipeline processing.

    metadata dict can include source-level info:
        author:        who wrote/owns this document
        last_modified: ISO date string (e.g. "2026-04-15")
        created:       ISO date string
        version:       version number (int or str)
        status:        "draft" | "published" | "archived"
        labels:        list of tags/labels
        (any other key-value pairs the source provides)
    """
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _normalize_input(input_data: Union[str, list[dict], dict]) -> list[_Document]:
    """Normalize input to a list of documents.

    Accepts:
        - str: single document text
        - dict: {"id", "title", "text", optional "metadata"}
        - list[dict]: multiple documents
    """
    if isinstance(input_data, str):
        return [_Document(doc_id="doc_1", title="", text=input_data)]

    if isinstance(input_data, dict):
        return [_Document(
            doc_id=str(input_data.get("id", "doc_1")),
            title=str(input_data.get("title", "")),
            text=str(input_data.get("text", "")),
            metadata=input_data.get("metadata", {}),
        )]

    if isinstance(input_data, list):
        docs = []
        for i, item in enumerate(input_data):
            if isinstance(item, str):
                docs.append(_Document(doc_id=f"doc_{i+1}", title="", text=item))
            elif isinstance(item, dict):
                docs.append(_Document(
                    doc_id=str(item.get("id", f"doc_{i+1}")),
                    title=str(item.get("title", "")),
                    text=str(item.get("text", "")),
                    metadata=item.get("metadata", {}),
                ))
        return docs

    raise ValueError(f"Unsupported input type: {type(input_data)}")


# =====================================================================
# Pipeline
# =====================================================================

class Pipeline:
    """AIQ Pipeline — runs the full quality pipeline on one or more documents.

    Phase 1 (A10-A14): runs PER DOCUMENT — each document gets its own chunks
    Phase 2 (A22):     runs on pooled chunks
    Phase 3 (A30-A32): runs on pooled chunks — enables cross-document detection
    """

    def __init__(self, config: Optional[AIQConfig] = None):
        self.config = config or AIQConfig()

    def run(self, input_data: Union[str, list[dict], dict]) -> PipelineResult:
        """Run the pipeline on one or more documents.

        Args:
            input_data: document text (str), or dict with id/title/text,
                        or list of dicts for multi-document

        Returns:
            PipelineResult with chunks, findings, and metrics
        """
        t0 = time.perf_counter()
        cfg = self.config
        docs = _normalize_input(input_data)

        result = PipelineResult()
        all_raw_chunks: list[Chunk] = []
        all_pipeline_chunks: list[Chunk] = []
        all_sections = []

        # ── Phase 1: per-document intake ──
        if 1 in cfg.phases:
            from aiq.a10 import RawChunker, A10Config
            from aiq.a11 import DomainInferrer, A11Config
            from aiq.a12 import Normalizer, A12Config
            from aiq.a13 import Structurer, A13Config
            from aiq.a14 import SmartChunker, A14Config

            a10_config = A10Config(
                min_words=cfg.chunk_min_words,
                max_words=cfg.chunk_max_words,
                strip_html=cfg.strip_html,
            )
            a12_config = A12Config(
                mode=cfg.phase1_mode,
                llm_call=cfg.llm_call,
                vision_call=cfg.vision_call,
                image_dir=cfg.image_dir,
            )
            a13_config = A13Config(mode=cfg.phase1_mode, llm_call=cfg.llm_call)
            a14_config = A14Config(
                min_words=cfg.chunk_min_words,
                max_words=cfg.chunk_max_words,
                topic_shift_threshold=cfg.topic_shift_threshold,
            )

            # A11 runs on ALL documents combined (one domain context)
            # First, get raw chunks from all docs for A11
            a10 = RawChunker(config=a10_config)
            temp_all_chunks = []
            per_doc_raw = {}

            for doc in docs:
                a10_out = a10.run(doc.text, source_ref=doc.title or doc.doc_id)
                # Tag chunks with source document info + metadata
                for chunk in a10_out.chunks:
                    chunk.source_page_id = doc.doc_id
                    chunk.source_page_title = doc.title
                    if doc.metadata:
                        chunk.metadata.update(doc.metadata)
                all_raw_chunks.extend(a10_out.chunks)
                temp_all_chunks.extend(a10_out.chunks)
                per_doc_raw[doc.doc_id] = a10_out

            result.module_outputs["A10"] = ModuleOutput(
                module_id="A10", module_name="Raw Chunking",
                detected=0, resolved=0, remaining=0,
                words_in=sum(o.words_in for o in per_doc_raw.values()),
                words_out=sum(o.words_out for o in per_doc_raw.values()),
                chunks=all_raw_chunks,
            )

            # A11: domain context from all chunks combined
            a11_config = A11Config(
                mode=cfg.phase1_mode,
                llm_call=cfg.llm_call,
                source_title=docs[0].title if len(docs) == 1 else "",
            )
            a11 = DomainInferrer(config=a11_config)
            a11_out = a11.run(temp_all_chunks)
            domain_ctx = a11_out.data["domain_context"]

            # Apply user overrides
            if cfg.domain_type:
                domain_ctx.domain_type = cfg.domain_type
            if cfg.company_name:
                domain_ctx.company_name = cfg.company_name
            if cfg.extra_actors:
                domain_ctx.actors.update(cfg.extra_actors)
            if cfg.extra_acronyms:
                domain_ctx.acronyms.update(cfg.extra_acronyms)
            if cfg.extra_products:
                domain_ctx.product_names.extend(cfg.extra_products)
            if cfg.extra_destructive:
                domain_ctx.destructive_patterns.extend(cfg.extra_destructive)

            result.domain_context = domain_ctx
            result.module_outputs["A11"] = a11_out

            # A12-A14: per document
            a12 = Normalizer(config=A12Config(
                mode=cfg.phase1_mode, llm_call=cfg.llm_call,
                vision_call=cfg.vision_call,
                domain_type=domain_ctx.domain_type,
                image_dir=cfg.image_dir,
            ))
            a13 = Structurer(config=a13_config)
            a14 = SmartChunker(config=a14_config)

            all_a12_findings = []
            all_a13_sections = []

            for doc in docs:
                # A12: normalize
                a12_out = a12.run(doc.text, source_ref=doc.title or doc.doc_id)
                normalized_html = a12_out.data.get("normalized_html", doc.text)
                all_a12_findings.extend(a12_out.findings)

                # A13: structure
                a13_out = a13.run(normalized_html, source_ref=doc.title or doc.doc_id)
                sections = a13_out.data.get("sections", a13_out.findings)
                all_a13_sections.extend(sections)

                # A14: smart chunk
                a14_out = a14.run(sections, source_ref=doc.title or doc.doc_id)

                # Tag chunks with source document info + metadata
                for chunk in a14_out.chunks:
                    chunk.source_page_id = doc.doc_id
                    chunk.source_page_title = doc.title
                    if doc.metadata:
                        chunk.metadata.update(doc.metadata)

                all_pipeline_chunks.extend(a14_out.chunks)

                result.documents.append({
                    "id": doc.doc_id,
                    "title": doc.title,
                    "raw_chunks": len(per_doc_raw[doc.doc_id].chunks),
                    "pipeline_chunks": len(a14_out.chunks),
                })

            result.module_outputs["A12"] = ModuleOutput(
                module_id="A12", module_name="Normalize",
                detected=len(all_a12_findings),
                findings=all_a12_findings,
            )
            result.module_outputs["A14"] = ModuleOutput(
                module_id="A14", module_name="Smart Chunker",
                chunks=all_pipeline_chunks,
            )

        # ── Phase 2: metadata enrichment (pooled) ──
        if 2 in cfg.phases and all_pipeline_chunks:
            try:
                from aiq.a22 import MetadataEnricher, A22Config
                a22_config = A22Config(
                    flag_stale=cfg.freshness_threshold_days > 0,
                    stale_months=max(1, cfg.freshness_threshold_days // 30),
                )
                a22 = MetadataEnricher(config=a22_config)
                a22_out = a22.run(all_pipeline_chunks)
                result.module_outputs["A22"] = a22_out
            except ImportError:
                pass  # A22 not available

        # ── Phase 3: detection & fix (pooled — cross-document) ──
        if 3 in cfg.phases and all_pipeline_chunks:
            domain_ctx = result.domain_context or DomainContext()

            # A30: Semantic Clarity
            from aiq.a30 import ClarityChecker, A30Config
            a30_config = A30Config(
                pronoun_mode=cfg.clarity_pronoun_mode,
                acronym_mode=cfg.clarity_acronym_mode,
                llm_call=cfg.llm_call,
            )
            a30 = ClarityChecker(config=a30_config)
            a30_out = a30.run(all_pipeline_chunks, domain_context=domain_ctx)
            result.module_outputs["A30"] = a30_out

            # A31: Content Governance
            from aiq.a31 import Classifier, A31Config
            a31_config = A31Config(
                pii_mode=cfg.pii_mode,
                llm_call=cfg.llm_call,
            )
            a31 = Classifier(config=a31_config)
            a31_out = a31.run(all_pipeline_chunks, domain_context=domain_ctx)
            result.module_outputs["A31"] = a31_out

            # Custom rules — apply after A31, only on chunks still tagged as CONTENT
            if cfg.custom_rules:
                import re as _re
                custom_count = 0
                for chunk in all_pipeline_chunks:
                    if chunk.tag != ChunkTag.CONTENT:
                        continue  # built-in tag takes priority
                    for rule in cfg.custom_rules:
                        pattern = rule.get("pattern", "")
                        action = rule.get("action", "review")
                        reason = rule.get("reason", f"Custom rule: {pattern}")
                        if pattern and _re.search(pattern, chunk.content, _re.IGNORECASE):
                            chunk.tag = ChunkTag.CUSTOM_BLOCK if action == "block" else ChunkTag.CUSTOM_REVIEW
                            chunk.tag_reason = reason
                            chunk.tag_module = "custom_rule"
                            custom_count += 1
                            break  # first matching rule wins per chunk
                if custom_count:
                    result.module_outputs["custom_rules"] = ModuleOutput(
                        module_id="custom", module_name="Custom Rules",
                        detected=custom_count, resolved=custom_count,
                    )

            # A32: Consistency (cross-document!)
            from aiq.a32 import ConsistencyChecker, A32Config
            a32_config = A32Config(
                use_llm_judge=cfg.consistency_llm_judge,
                llm_call=cfg.llm_call,
            )
            a32 = ConsistencyChecker(config=a32_config)
            a32_out = a32.run(all_pipeline_chunks, domain_context=domain_ctx)
            result.module_outputs["A32"] = a32_out

        # ── Post-Phase 3: Re-chunk cleaned content ──
        # After A30/A31/A32 have edited content (rewrites, removals, resolutions),
        # re-chunk to: remove empties, fix chunk boundaries, add enrichment prefix.
        if 3 in cfg.phases and all_pipeline_chunks:
            from aiq.a14 import SmartChunker, A14Config

            # 1. Remove empty chunks (content fully removed by A31)
            pre_clean_count = len(all_pipeline_chunks)
            all_pipeline_chunks = [c for c in all_pipeline_chunks if c.words >= 10]
            removed_empty = pre_clean_count - len(all_pipeline_chunks)

            # 2. Re-chunk: rebuild sections from cleaned chunks, re-run A14
            from aiq.a13.structurer import Section
            cleaned_sections = []
            for chunk in all_pipeline_chunks:
                cleaned_sections.append(Section(
                    section_id=chunk.chunk_id,
                    heading=chunk.heading,
                    heading_level=2,
                    content=chunk.content,
                    words=chunk.words,
                ))

            a14_config = A14Config(
                min_words=cfg.chunk_min_words,
                max_words=cfg.chunk_max_words,
                topic_shift_threshold=cfg.topic_shift_threshold,
            )
            a14_rechunk = SmartChunker(config=a14_config)
            rechunk_out = a14_rechunk.run(cleaned_sections, source_ref="rechunk")

            # 3. Add enrichment prefix: prepend heading + key topic words to content
            for chunk in rechunk_out.chunks:
                if chunk.heading and not chunk.content.startswith(chunk.heading):
                    # Extract top topic keywords from content
                    import re as _re
                    topic_words = _re.findall(r'\b[a-z]{4,}\b', chunk.content.lower())
                    _enrich_stop = {
                        "the", "and", "for", "with", "from", "this", "that",
                        "are", "was", "has", "have", "will", "can", "our",
                        "your", "all", "also", "not", "been", "may", "must",
                    }
                    from collections import Counter as _Counter
                    freq = _Counter(w for w in topic_words if w not in _enrich_stop)
                    top_keywords = [w for w, _ in freq.most_common(5)]
                    keyword_str = ", ".join(top_keywords) if top_keywords else ""

                    prefix = chunk.heading
                    if keyword_str:
                        prefix += f". {keyword_str}."
                    else:
                        prefix += "."

                    chunk.content = f"{prefix} {chunk.content}"
                    chunk.words = len(chunk.content.split())

            # Restore source metadata from original chunks
            orig_meta = {}
            for c in all_pipeline_chunks:
                orig_meta[c.chunk_id] = {
                    "source_page_id": c.source_page_id,
                    "source_page_title": c.source_page_title,
                    "metadata": dict(c.metadata),
                }

            # Apply metadata to rechunked chunks (use first original chunk's metadata)
            if orig_meta:
                first_meta = list(orig_meta.values())[0]
                for chunk in rechunk_out.chunks:
                    chunk.source_page_id = first_meta.get("source_page_id", "")
                    chunk.source_page_title = first_meta.get("source_page_title", "")

            all_pipeline_chunks = rechunk_out.chunks

            result.module_outputs["A14_rechunk"] = ModuleOutput(
                module_id="A14_rechunk", module_name="Re-chunk (post-clean)",
                detected=removed_empty,
                resolved=removed_empty,
                remaining=0,
                data={
                    "chunks_before_clean": pre_clean_count,
                    "empty_removed": removed_empty,
                    "chunks_after_rechunk": len(all_pipeline_chunks),
                },
            )

        # ── Build result ──
        result.raw_chunks = all_raw_chunks
        result.chunks = all_pipeline_chunks
        result.elapsed_seconds = time.perf_counter() - t0

        # Aggregate counts
        for mid, mout in result.module_outputs.items():
            result.total_detected += mout.detected
            result.total_resolved += mout.resolved
            result.total_remaining += mout.remaining

        return result

    def evaluate(self, pipeline_result: PipelineResult) -> PipelineResult:
        """Run Phase 4 evaluation on a pipeline result.

        Generates Q&A pairs (A41), runs retrieval tests (A42),
        and computes metrics (A43). Separate from run() because
        evaluation is optional and may require LLM/embeddings.

        Args:
            pipeline_result: output from Pipeline.run()

        Returns:
            Updated PipelineResult with A41/A42/A43 outputs added
        """
        cfg = self.config
        chunks = pipeline_result.chunks
        raw_chunks = pipeline_result.raw_chunks
        domain_ctx = pipeline_result.domain_context

        if not chunks:
            return pipeline_result

        # ── A41: Q&A Generation ──
        from aiq.a41 import QAGenerator, A41Config
        a41_config = A41Config(
            total_questions=cfg.eval_total_questions,
            llm_call=cfg.llm_call,
            domain_type=domain_ctx.domain_type if domain_ctx else "",
            user_qa_pairs=cfg.user_qa_pairs,
        )
        a41 = QAGenerator(config=a41_config)

        # Gather Phase 3 findings for targeted question generation
        a30_findings = getattr(pipeline_result.module_outputs.get("A30"), "findings", [])
        a31_findings = getattr(pipeline_result.module_outputs.get("A31"), "findings", [])
        a32_findings = getattr(pipeline_result.module_outputs.get("A32"), "findings", [])

        a41_out = a41.run(
            chunks,
            a30_findings=a30_findings,
            a31_findings=a31_findings,
            a32_findings=a32_findings,
            domain_context=domain_ctx,
        )
        pipeline_result.module_outputs["A41"] = a41_out
        qa_pairs = a41_out.data["qa_set"].pairs

        if not qa_pairs:
            return pipeline_result

        # ── A42: Retrieval Test ──
        from aiq.a42 import RetrievalTester, A42Config
        a42_config = A42Config(
            retrieval_method="bm25",  # default: no embedding dependency
            judge_method="cosine",     # default: no LLM dependency
            llm_call=cfg.llm_call,
        )
        a42 = RetrievalTester(config=a42_config)
        a42_out = a42.run(qa_pairs, raw_chunks, chunks)
        pipeline_result.module_outputs["A42"] = a42_out

        # ── A43: Metrics ──
        from aiq.a43 import MetricsCalculator, A43Config
        a43_config = A43Config()
        a43 = MetricsCalculator(config=a43_config)

        pair_results = a42_out.data["test_result"].pair_results
        session_modules = {
            f"a{k.lower().lstrip('a')}_output" if k[0] == "A" else k: v
            for k, v in pipeline_result.module_outputs.items()
        }
        a43_out = a43.run(pair_results, session_modules=session_modules, chunks=chunks)
        pipeline_result.module_outputs["A43"] = a43_out

        return pipeline_result


# Backward compatibility alias
PipelineConfig = AIQConfig
