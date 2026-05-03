# AIQ: Improving Knowledge Base Quality to Optimize Retrieval in RAG Systems

AIQ is a modular, data-centric pipeline that detects and resolves content quality issues in knowledge bases **before** they enter the retrieval layer. It works with any RAG system — you improve the data, retrieval improves automatically.

## The Problem

Enterprise knowledge bases contain:
- **PII** (personal emails, phone numbers, account numbers)
- **Internal notes** ("agents only — do not share with customers")
- **Placeholders** (TODO, TBD, coming soon)
- **Contradictions** ("refunds take 5 days" vs "refunds take 10 days")
- **Vague content** ("the team handles most edge cases")
- **Stale content** (outdated policies still appearing in search)

A better embedding model finds the PII-containing chunk **more accurately**. A faster vector database serves the contradictory answer **more quickly**. The problem is upstream: **the data itself is not ready for retrieval**.

## Quick Start

```bash
pip install aiq
```

```python
from aiq import analyze

# Rule-based analysis — no LLM needed, no API key required
result = analyze("your document text or HTML")

print(f"Issues found: {result.total_detected}")
print(f"Issues resolved: {result.total_resolved}")

# See what was detected
for chunk in result.chunks:
    if chunk.tag.value != "content":
        print(f"  [{chunk.tag.value}] {chunk.tag_reason}")
```

### With LLM (better quality)

```python
from aiq import analyze, AIQConfig

config = AIQConfig(
    llm_provider="openai",      # or "groq", "anthropic"
    llm_api_key="sk-...",
)
result = analyze("your text", config=config)
```

### From Files

```python
from aiq import analyze
from aiq.loader import load_file, load_directory

# Single file (HTML, DOCX, PDF, text, Markdown)
result = analyze(load_file("refund_policy.docx"))

# Entire directory
result = analyze(load_directory("knowledge_base/"))
```

### Multiple Documents (cross-document detection)

```python
docs = [
    {"id": "page_1", "title": "Refund Policy", "text": "Refunds take 5 days..."},
    {"id": "page_2", "title": "Billing FAQ", "text": "Refunds take 10 days..."},
]
result = analyze(docs)
# AIQ detects the contradiction between page_1 and page_2
```

## What AIQ Detects

| Category | Examples | Action |
|----------|----------|--------|
| **PII** | Emails, phone numbers, names, account numbers | Block |
| **Internal notes** | "INTERNAL NOTE", "do not share", "agents only" | Block |
| **Placeholders** | TODO, TBD, FIXME, "[insert link]" | Block |
| **Editorial artifacts** | HTML comments, tracked changes, draft markers | Block |
| **Metadata leaks** | Jira IDs, Slack channels, internal URLs | Block |
| **Vague claims** | "handles most cases", "industry leading" | Review |
| **Destructive actions** | "delete account", "purge data" | Review |
| **Broken references** | "see FAQ", "image not available" | Review |
| **Contradictions** | Same fact, different values across documents | Review |
| **Stale content** | Outdated dates, old version references | Review |
| **Undefined acronyms** | Acronyms not expanded in the chunk | Auto-fix |
| **Ambiguous pronouns** | "They handle escalations" (who?) | Auto-fix |

## Pipeline Phases

```
Document(s) --> Phase 1: Intake & Structure (A10-A14)
                  Smart chunking, heading detection, table/figure extraction

            --> Phase 2: Metadata Enrichment (A22)
                  Date extraction, version tracking, staleness detection

            --> Phase 3: Detection & Fix (A30-A32)
                  Clarity fixes, governance tagging, consistency checking

            --> Phase 4: Evaluation (A41-A43) [optional]
                  Q&A generation, retrieval testing, before/after metrics
```

**Every module works without LLM.** LLM is an optional enhancement that improves quality.

## Configuration

```python
from aiq import AIQConfig

config = AIQConfig(
    # LLM (optional)
    llm_provider="openai",
    llm_api_key="sk-...",

    # PII detection sensitivity
    pii_mode="smart",             # "strict" | "smart" | "lenient"

    # Detection confidence threshold
    detection_confidence="medium", # "high" | "medium" | "low" or 0.0-1.0

    # Tag behavior overrides
    tag_behavior={
        "destructive": "block",   # upgrade from "review" to "block"
        "vague_claim": "allow",   # downgrade from "review" to "allow"
    },

    # Custom detection rules
    custom_rules=[
        {"pattern": "beta feature", "action": "review", "reason": "Beta content"},
        {"pattern": "competitor", "action": "block", "reason": "Competitor mention"},
    ],

    # Content freshness
    freshness_threshold_days=180,  # flag content older than 6 months

    # Domain knowledge overrides
    domain_type="medical",
    extra_acronyms={"MRN": "Medical Record Number"},
)
```

## Evaluation (Optional)

```python
from aiq import Pipeline, AIQConfig

pipeline = Pipeline(AIQConfig())
result = pipeline.run("your documents")

# Run Phase 4: generates test Q&A, runs retrieval tests, computes metrics
eval_result = pipeline.evaluate(result)

metrics = eval_result.module_outputs["A43"].data["metrics_result"]
print(f"Recall before AIQ: {metrics.recall_before.rate:.0%}")
print(f"Recall after AIQ:  {metrics.recall_after.rate:.0%}")
```

## Installation Options

```bash
pip install aiq              # Core (HTML, text, Markdown)
pip install aiq[docx]        # + DOCX support
pip install aiq[pdf]         # + PDF support
pip install aiq[openai]      # + OpenAI LLM
pip install aiq[all]         # Everything
```

## Architecture

AIQ is organized as 12 independent modules across 4 phases. Each module:
- Has a standard contract: input chunks, output `ModuleOutput` with `(detected, resolved, remaining)`
- Works standalone: `from aiq.a31 import Classifier; findings = Classifier().run(chunks)`
- Has rule-based fallback: no LLM required for any module
- Tags content, never deletes: original content is always preserved

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - see [LICENSE](LICENSE)
