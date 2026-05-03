"""AIQ Quick Start — analyze a knowledge base document in 5 lines."""

from aiq import analyze

result = analyze("""
<h1>Refund Policy</h1>
<p>All refunds are processed within 5 business days by the billing team.
Enterprise customers get priority processing with a 2 business day SLA.</p>

<p>For questions, contact John Smith at john.smith@acme.com or (555) 867-5309.</p>

<p>INTERNAL NOTE: The actual SLA is 48 hours but we tell customers 5 days as a buffer.
Do not share this with customers.</p>

<p>TODO: Add information about the new refund portal launching Q2 2026.</p>

<h2>Payment Methods</h2>
<p>We accept Visa, Mastercard, and bank transfers.
The system seamlessly handles most payment edge cases.</p>
""")

print(f"Chunks analyzed: {len(result.chunks)}")
print(f"Issues found:    {result.total_detected}")
print(f"Issues resolved: {result.total_resolved}")
print(f"Domain:          {result.domain_context.domain_type}")
print()

for chunk in result.chunks:
    tag = chunk.tag.value
    if tag != "content":
        print(f"  [{tag.upper()}] {chunk.tag_reason}")
