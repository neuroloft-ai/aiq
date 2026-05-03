"""AIQ Custom Rules Example — add your own detection patterns."""

from aiq import analyze, AIQConfig

config = AIQConfig(
    # Custom regex patterns — block or flag content your organization cares about
    custom_rules=[
        {"pattern": r"beta\s+feature", "action": "review", "reason": "Beta content needs review before serving"},
        {"pattern": r"competitor|competing\s+product", "action": "block", "reason": "Competitor mentions should not be served"},
        {"pattern": r"deprecated|end.of.life", "action": "block", "reason": "Deprecated content"},
        {"pattern": r"price:\s*\$\d+", "action": "review", "reason": "Pricing needs approval before serving"},
        {"pattern": r"john|jane|mike", "action": "block", "reason": "Employee names detected"},
    ],

    # Override default tag behaviors
    tag_behavior={
        "destructive": "block",     # upgrade: was "review", now "block"
        "vague_claim": "allow",     # downgrade: was "review", now "allow"
        "stale": "block",           # upgrade: was "review", now "block"
    },

    # Detection confidence — only act on high-confidence findings
    detection_confidence="high",
)

text = """
Our new beta feature allows bulk exports of customer data.
The deprecated v1 API endpoint will be removed next quarter.
Unlike competitor Acme Corp, our platform offers unlimited storage.
For pricing details, see: price: $99/month for Enterprise.
Contact John Smith for technical questions.
The system seamlessly handles most edge cases automatically.
"""

result = analyze(text, config=config)

print("Detection results:")
for chunk in result.chunks:
    tag = chunk.tag.value
    if tag != "content":
        behavior = chunk.tag.behavior(config.tag_behavior)
        print(f"  [{behavior.upper()}] {tag}: {chunk.tag_reason}")
    else:
        print(f"  [SAFE] No issues detected")
