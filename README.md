# AIQ: Improving Knowledge Base Quality to Optimize Retrieval in RAG Systems

AIQ is a modular, data-centric pipeline that detects and resolves content quality issues in knowledge bases **before** they enter the retrieval layer. It works with any RAG system — you improve the data, retrieval improves automatically.

**No LLM required.** Everything works rule-based out of the box. LLM is an optional enhancement.

## Install

```bash
pip install git+https://github.com/neuroloft-ai/aiq.git
```

## Complete Working Example (copy-paste this)

```python
from aiq import analyze, AIQConfig

# Analyze a document — works with plain text, HTML, or multiple documents
result = analyze("""
<h1>Refund Policy</h1>
<p>All refunds are processed within 5 business days by the billing team.</p>
<p>Contact Sarah Johnson at sarah.johnson@acme.com or (555) 867-5309.</p>
<p>INTERNAL NOTE: The actual SLA is 48 hours. Do not share with customers.</p>
<p>TODO: Add pricing for Enterprise tier.</p>
""")

# What was found
print(f"Chunks:   {len(result.chunks)}")
print(f"Issues:   {result.total_detected}")
print(f"Domain:   {result.domain_context.domain_type}")

# See each chunk and its safety tag
for chunk in result.chunks:
    tag = chunk.tag.value           # "content", "pii", "internal_only", "placeholder", etc.
    behavior = chunk.tag.default_behavior  # "answer", "block", or "review"
    if tag != "content":
        print(f"  [{behavior.upper()}] {tag}: {chunk.tag_reason}")
    else:
        print(f"  [SAFE] {chunk.content[:80]}")

# Access specific module results
a31 = result.module_outputs.get("A31")  # Content Governance
if a31:
    for finding in a31.findings:
        print(f"  Finding: [{finding.tag.value}] {finding.reason}")
```

## With LLM (better quality, optional)

```python
from aiq import analyze, AIQConfig

config = AIQConfig(
    llm_provider="openai",      # or "groq", "anthropic"
    llm_api_key="sk-...",       # your API key
    llm_model="gpt-4o-mini",    # optional, uses provider default
)
result = analyze("your document text", config=config)
```

Or bring your own LLM function:

```python
config = AIQConfig(
    llm_call=lambda prompt: my_llm(prompt),  # any callable that returns str
)
```

## Input Formats

### Plain text or HTML string

```python
result = analyze("your document text or HTML")
```

### From files (HTML, DOCX, PDF, text, Markdown)

```python
from aiq.loader import load_file, load_directory

result = analyze(load_file("policy.docx"))
result = analyze(load_directory("kb_pages/"))    # all files in a directory
```

### Multiple documents with metadata

```python
docs = [
    {
        "id": "page_1",
        "title": "Refund Policy",
        "text": "Refunds take 5 business days...",
        "metadata": {                              # optional
            "author": "Product Team",
            "last_modified": "2026-04-15",
            "status": "published",
        },
    },
    {
        "id": "page_2",
        "title": "Billing FAQ",
        "text": "Refunds take 10 business days...",
    },
]
result = analyze(docs)
# AIQ detects contradictions across documents
```

## Result Object

```python
result = analyze(document)

# Top-level
result.chunks                  # list[Chunk] — processed chunks with safety tags
result.raw_chunks              # list[Chunk] — original chunks before processing
result.domain_context          # DomainContext — detected domain, actors, acronyms
result.total_detected          # int — total issues found
result.total_resolved          # int — total issues fixed
result.elapsed_seconds         # float — processing time
result.documents               # list[dict] — per-document info (multi-doc mode)
result.module_outputs          # dict — per-module results (A10, A11, ..., A32)

# Each chunk
chunk.chunk_id                 # str — unique ID
chunk.heading                  # str — section heading
chunk.content                  # str — chunk text
chunk.words                    # int — word count
chunk.tag                      # ChunkTag — safety classification
chunk.tag.value                # str — "content", "pii", "internal_only", etc.
chunk.tag.default_behavior     # str — "answer", "block", or "review"
chunk.tag_reason               # str — why this tag was assigned
chunk.source_page_id           # str — which document this chunk came from
chunk.metadata                 # dict — author, last_modified, etc.
```

## What AIQ Detects

| Category | Examples | Default Action |
|----------|----------|----------------|
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

## Configuration Options

```python
config = AIQConfig(
    # ── LLM (optional — everything works without) ──
    llm_provider="openai",           # "openai" | "groq" | "anthropic"
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",         # uses provider default if empty

    # ── PII detection sensitivity ──
    pii_mode="smart",                # "strict" (block all) | "smart" (skip functional) | "lenient" (names+emails only)

    # ── Detection confidence threshold ──
    detection_confidence="medium",   # "high" (0.8) | "medium" (0.5) | "low" (0.3) or any float 0.0-1.0

    # ── Tag behavior overrides ──
    tag_behavior={
        "destructive": "block",      # upgrade from "review" to "block"
        "vague_claim": "allow",      # downgrade from "review" to "allow"
    },

    # ── Custom detection rules (your own regex patterns) ──
    custom_rules=[
        {"pattern": "beta feature", "action": "review", "reason": "Beta content needs review"},
        {"pattern": "competitor|competing", "action": "block", "reason": "Competitor mention"},
        {"pattern": "deprecated", "action": "block", "reason": "Deprecated content"},
    ],

    # ── Content freshness ──
    freshness_threshold_days=180,    # flag content older than 6 months (0 = disabled)

    # ── Domain knowledge (override auto-detection) ──
    domain_type="medical",           # override detected domain
    company_name="Acme Health",      # override detected company
    extra_acronyms={"MRN": "Medical Record Number"},  # add to detected acronyms
    extra_actors={"radiology": "radiology"},           # add to detected teams

    # ── Chunking ──
    chunk_min_words=50,
    chunk_max_words=200,
)
```

## Evaluation (Optional — Phase 4)

```python
from aiq import Pipeline, AIQConfig

pipeline = Pipeline(AIQConfig(
    user_qa_pairs=[
        {"question": "How long do refunds take?", "expected_answer": "5 business days", "expected_behavior": "answer"},
        {"question": "What is the internal SLA?", "expected_answer": "", "expected_behavior": "block"},
    ],
))

# Phase 1-3: process documents
result = pipeline.run("your documents")

# Phase 4: generate Q&A, test retrieval, compute metrics
eval_result = pipeline.evaluate(result)

metrics = eval_result.module_outputs["A43"].data["metrics_result"]
print(f"Recall before AIQ: {metrics.recall_before.rate:.0%}")
print(f"Recall after AIQ:  {metrics.recall_after.rate:.0%}")
```

## Using Individual Modules Directly

```python
from aiq.a10 import RawChunker
from aiq.a11 import DomainInferrer
from aiq.a31 import Classifier, A31Config

# Chunk a document
chunks = RawChunker().run("your text").chunks

# Detect domain
ctx = DomainInferrer().run(chunks).data["domain_context"]

# Run governance detection
result = Classifier(A31Config(pii_mode="strict")).run(chunks, domain_context=ctx)
for finding in result.findings:
    print(f"  [{finding.tag.value}] {finding.reason}")
```

## Pipeline Phases

```
Document(s) --> Phase 1: Intake & Structure (A10-A14)
                  A10: Raw chunking (baseline)
                  A11: Domain intelligence (actors, acronyms, scope)
                  A12: Content normalization (tables, figures, procedures)
                  A13: Structure & headings (orphan/generic heading fix)
                  A14: Smart chunking (topic-aware splitting)

            --> Phase 2: Metadata Enrichment (A22)
                  Date extraction, version tracking, staleness detection

            --> Phase 3: Detection & Fix (A30-A32)
                  A30: Semantic clarity (pronouns, acronyms, vague refs)
                  A31: Content governance (PII, internal, placeholders, 9 detectors)
                  A32: Consistency (numeric, authority, process contradictions)

            --> Phase 4: Evaluation (A41-A43) [optional]
                  A41: Q&A generation from pipeline results
                  A42: Retrieval testing (before vs after)
                  A43: Metrics (recall, safety, risk scores)
```

Every module works without LLM. LLM is an optional enhancement that improves quality.

## Install Options

```bash
pip install git+https://github.com/neuroloft-ai/aiq.git           # core
pip install "aiq[openai] @ git+https://github.com/neuroloft-ai/aiq.git"  # + OpenAI
pip install "aiq[all] @ git+https://github.com/neuroloft-ai/aiq.git"     # everything
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding detectors, file formats, and LLM providers.

## License

Apache 2.0 — see [LICENSE](LICENSE)
