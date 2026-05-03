"""AIQ Multi-Document Example — detect cross-document contradictions."""

from aiq import analyze, AIQConfig

docs = [
    {
        "id": "refund_policy",
        "title": "Refund Policy",
        "text": """
            <h1>Refund Policy</h1>
            <p>All refunds are processed within 5 business days.
            The billing team handles all refund requests and escalations.</p>
        """,
        "metadata": {
            "author": "Product Team",
            "last_modified": "2026-04-15",
            "status": "published",
        },
    },
    {
        "id": "billing_faq",
        "title": "Billing FAQ",
        "text": """
            <h1>Billing FAQ</h1>
            <p>Refunds typically take 10 business days to process.
            The finance team handles all refund requests.</p>
            <p>[AGENTS ONLY] If customer pushes back, offer a 50% partial refund immediately.</p>
        """,
        "metadata": {
            "author": "Support Intern",
            "last_modified": "2024-06-01",
            "status": "draft",
        },
    },
]

config = AIQConfig(
    pii_mode="smart",
    freshness_threshold_days=365,  # flag content older than 1 year
    tag_behavior={"destructive": "block"},
    custom_rules=[
        {"pattern": "partial refund", "action": "review", "reason": "Partial refund policy"},
    ],
)

result = analyze(docs, config=config)

print(f"Documents:    {len(result.documents)}")
print(f"Total chunks: {len(result.chunks)}")
print(f"Detected:     {result.total_detected}")
print(f"Domain:       {result.domain_context.domain_type}")
print()

print("Chunk results:")
for chunk in result.chunks:
    tag = chunk.tag.value
    behavior = chunk.tag.behavior(config.tag_behavior)
    page = chunk.source_page_id
    author = chunk.metadata.get("author", "unknown")
    if tag != "content":
        print(f"  [{behavior.upper()}] {page} (by {author}): {chunk.tag_reason}")
    else:
        print(f"  [OK] {page} (by {author}): safe to serve")
