"""Test each AIQ module individually.

Tests each module in isolation with known inputs and expected outputs.
Ordered by pipeline execution: Config → Loader → A10 → A11 → A12 → A13 → A14 → A22 → A30 → A31 → A32.
"""
import pytest
import os

# Load the Neuroloft KB once for all tests
KB_PATH = os.path.join(os.path.dirname(__file__), '..', 'examples', 'neuroloft_kb.html')
with open(KB_PATH, encoding='utf-8') as _f:
    NEUROLOFT_HTML = _f.read()


# =====================================================================
# Config
# =====================================================================

class TestConfig:
    def test_defaults(self):
        from aiq.pipeline import AIQConfig
        c = AIQConfig()
        assert c.pii_mode == 'smart'
        assert c.chunk_min_words == 50
        assert c.chunk_max_words == 200
        assert c.detection_confidence == 0.5
        assert c.llm_call is None

    def test_confidence_categorical(self):
        from aiq.pipeline import AIQConfig
        assert AIQConfig(detection_confidence='high').detection_confidence == 0.8
        assert AIQConfig(detection_confidence='medium').detection_confidence == 0.5
        assert AIQConfig(detection_confidence='low').detection_confidence == 0.3

    def test_confidence_numeric(self):
        from aiq.pipeline import AIQConfig
        assert AIQConfig(detection_confidence=0.65).detection_confidence == 0.65


# =====================================================================
# Loader
# =====================================================================

class TestLoader:
    def test_load_html_file(self):
        from aiq.loader import load_file
        doc = load_file(KB_PATH)
        assert doc['id'] == 'neuroloft_kb.html'
        assert len(doc['text']) > 1000
        assert 'source_path' in doc.get('metadata', {})

    def test_load_nonexistent_raises(self):
        from aiq.loader import load_file
        with pytest.raises(FileNotFoundError):
            load_file('nonexistent_file.html')

    def test_normalize_string(self):
        from aiq.pipeline import _normalize_input
        docs = _normalize_input('Hello world')
        assert len(docs) == 1
        assert docs[0].text == 'Hello world'

    def test_normalize_dict(self):
        from aiq.pipeline import _normalize_input
        docs = _normalize_input({'id': 'p1', 'title': 'T', 'text': 'Content'})
        assert docs[0].doc_id == 'p1'
        assert docs[0].title == 'T'

    def test_normalize_list(self):
        from aiq.pipeline import _normalize_input
        docs = _normalize_input([
            {'id': 'a', 'text': 'A'},
            {'id': 'b', 'text': 'B'},
        ])
        assert len(docs) == 2


# =====================================================================
# A10 — Raw Chunker
# =====================================================================

class TestA10:
    def test_produces_chunks(self):
        from aiq.a10 import RawChunker, A10Config
        out = RawChunker(A10Config(min_words=50, max_words=200, strip_html=True)).run(NEUROLOFT_HTML)
        assert len(out.chunks) > 0

    def test_chunk_ids_sequential(self):
        from aiq.a10 import RawChunker, A10Config
        out = RawChunker(A10Config(strip_html=True)).run(NEUROLOFT_HTML)
        for i, c in enumerate(out.chunks):
            assert c.chunk_id == f'raw_{i+1}'

    def test_no_headings_assigned(self):
        from aiq.a10 import RawChunker, A10Config
        out = RawChunker(A10Config(strip_html=True)).run(NEUROLOFT_HTML)
        assert all(c.heading == '' for c in out.chunks)

    def test_empty_input(self):
        from aiq.a10 import RawChunker
        out = RawChunker().run('')
        assert len(out.chunks) == 0

    def test_html_stripped(self):
        from aiq.a10 import RawChunker, A10Config
        out = RawChunker(A10Config(strip_html=True)).run('<p>Hello <b>world</b></p>')
        if out.chunks:
            assert '<b>' not in out.chunks[0].content


# =====================================================================
# A11 — Domain Intelligence
# =====================================================================

class TestA11:
    @pytest.fixture(autouse=True)
    def setup(self):
        from aiq.a10 import RawChunker, A10Config
        from aiq.a11 import DomainInferrer, A11Config
        chunks = RawChunker(A10Config(strip_html=True)).run(NEUROLOFT_HTML).chunks
        self.ctx = DomainInferrer(A11Config(
            mode='rule_only', source_title='Payment & Billing'
        )).run(chunks).data['domain_context']

    def test_domain_type(self):
        assert self.ctx.domain_type == 'support'

    def test_finds_acronyms(self):
        assert len(self.ctx.acronyms) >= 2

    def test_finds_actors(self):
        assert len(self.ctx.actors) >= 1

    def test_confidence_positive(self):
        assert self.ctx.confidence > 0


# =====================================================================
# A12 — Content Normalizer
# =====================================================================

class TestA12:
    @pytest.fixture(autouse=True)
    def setup(self):
        from aiq.a12 import Normalizer, A12Config
        self.out = Normalizer(A12Config(mode='rule_only')).run(NEUROLOFT_HTML)

    def test_finds_tables(self):
        tables = [e for e in self.out.findings if e.element_type == 'table']
        assert len(tables) >= 3

    def test_row_level_linearization(self):
        tables = [e for e in self.out.findings if e.element_type == 'table']
        has_rows = any(len(t.row_texts) > 0 for t in tables)
        assert has_rows, 'At least one table should have row-level text'

    def test_table_heading_in_rows(self):
        tables = [e for e in self.out.findings if e.element_type == 'table' and e.row_texts]
        for t in tables:
            heading_word = t.source_ref.replace('Table: ', '').split()[0]
            assert any(heading_word in row for row in t.row_texts), \
                f'Table heading "{heading_word}" should appear in row text'

    def test_normalized_html_has_markers(self):
        assert 'aiq-table-row' in self.out.data['normalized_html']

    def test_finds_figures(self):
        figures = [e for e in self.out.findings if e.element_type == 'figure']
        assert len(figures) >= 1


# =====================================================================
# A13 — Structure & Headings
# =====================================================================

class TestA13:
    def test_produces_sections(self):
        from aiq.a12 import Normalizer, A12Config
        from aiq.a13 import Structurer, A13Config
        normalized = Normalizer(A12Config(mode='rule_only')).run(NEUROLOFT_HTML)
        out = Structurer(A13Config(mode='rule_only')).run(normalized.data['normalized_html'])
        sections = out.data['sections']
        assert len(sections) > 0
        assert all(s.heading for s in sections if s.words >= 10)


# =====================================================================
# A14 — Smart Chunker
# =====================================================================

class TestA14:
    @pytest.fixture(autouse=True)
    def setup(self):
        from aiq.a12 import Normalizer, A12Config
        from aiq.a13 import Structurer, A13Config
        from aiq.a14 import SmartChunker, A14Config
        normalized = Normalizer(A12Config(mode='rule_only')).run(NEUROLOFT_HTML)
        sections = Structurer(A13Config(mode='rule_only')).run(
            normalized.data['normalized_html']).data['sections']
        self.out = SmartChunker(A14Config(min_words=50, max_words=200)).run(sections)

    def test_produces_chunks(self):
        assert len(self.out.chunks) > 0

    def test_chunk_ids_sequential(self):
        for i, c in enumerate(self.out.chunks):
            assert c.chunk_id == f'c{i+1}'

    def test_all_have_headings(self):
        assert all(c.heading for c in self.out.chunks)

    def test_no_oversized_chunks(self):
        oversized = [c for c in self.out.chunks if c.words > 260]
        assert len(oversized) == 0

    def test_no_empty_chunks(self):
        assert all(c.words > 0 for c in self.out.chunks)


# =====================================================================
# A22 — Metadata Enrichment
# =====================================================================

class TestA22:
    def test_detects_dates(self):
        from aiq.a10 import RawChunker, A10Config
        from aiq.a22 import MetadataEnricher, A22Config
        chunks = RawChunker(A10Config(strip_html=True)).run(NEUROLOFT_HTML).chunks
        out = MetadataEnricher(A22Config(flag_stale=True)).run(chunks)
        dated = [c for c in chunks if c.metadata.get('a22_dates')]
        assert len(dated) > 0


# =====================================================================
# A30 — Semantic Clarity
# =====================================================================

class TestA30:
    def test_detects_issues(self):
        from aiq.a30 import ClarityChecker, A30Config
        from aiq.core.types import Chunk, DomainContext
        chunk = Chunk(chunk_id='t1', heading='Billing',
                      content='They handle all refunds. Contact them for help. The CRM tracks requests.',
                      words=12, source_type='text')
        ctx = DomainContext(domain_type='support', acronyms={'CRM': 'Customer Relationship Management'},
                           actors={'billing': 'billing'})
        out = ClarityChecker(A30Config(pronoun_mode='rule_fix', acronym_mode='rule_fix')).run([chunk], ctx)
        assert out.detected > 0

    def test_expands_acronyms(self):
        from aiq.a30 import ClarityChecker, A30Config
        from aiq.core.types import Chunk, DomainContext
        chunk = Chunk(chunk_id='t1', heading='Billing',
                      content='Submit via CRM and check the SLA for response times and billing details.',
                      words=13, source_type='text')
        ctx = DomainContext(acronyms={'CRM': 'Customer Relationship Management',
                                      'SLA': 'Service Level Agreement'})
        ClarityChecker(A30Config(acronym_mode='rule_fix')).run([chunk], ctx)
        assert 'Customer Relationship Management' in chunk.content


# =====================================================================
# A31 — Content Governance
# =====================================================================

class TestA31:
    def test_removes_pii(self):
        from aiq.a31 import Classifier, A31Config
        from aiq.core.types import Chunk
        chunk = Chunk(chunk_id='t1', heading='Cases',
                      content='Contact Sarah Johnson at sarah.j@acme.com for help with your billing inquiry today.',
                      words=14, source_type='text')
        Classifier(A31Config(pii_mode='strict')).run([chunk])
        assert 'sarah.j@acme.com' not in chunk.content

    def test_removes_internal_marker(self):
        from aiq.a31 import Classifier, A31Config
        from aiq.core.types import Chunk
        chunk = Chunk(chunk_id='t1', heading='Notes',
                      content='FOR INTERNAL USE ONLY. Contact the manager for approval on all refund escalations.',
                      words=14, source_type='text')
        Classifier(A31Config()).run([chunk])
        assert chunk.content == '' or 'FOR INTERNAL USE ONLY' not in chunk.content

    def test_removes_placeholder(self):
        from aiq.a31 import Classifier, A31Config
        from aiq.core.types import Chunk
        chunk = Chunk(chunk_id='t1', heading='Setup',
                      content='Account setup requires verification. TODO: Add SSO configuration details for enterprise customers.',
                      words=12, source_type='text')
        Classifier(A31Config()).run([chunk])
        assert 'TODO' not in chunk.content

    def test_detects_findings(self):
        from aiq.a31 import Classifier, A31Config
        from aiq.core.types import Chunk
        chunk = Chunk(chunk_id='t1', heading='Cases',
                      content='JIRA-9999 tracks the bug. Contact john@test.com for details about the billing issue.',
                      words=14, source_type='text')
        out = Classifier(A31Config(pii_mode='strict')).run([chunk])
        assert out.detected > 0


# =====================================================================
# A32 — Consistency Checking
# =====================================================================

class TestA32:
    def test_detects_numeric_contradiction(self):
        from aiq.a32 import ConsistencyChecker, A32Config
        from aiq.core.types import Chunk
        chunk_a = Chunk(chunk_id='a', heading='Refund Policy',
                        content='Refunds are processed within 5 business days from the approved request date.',
                        words=12, source_type='text')
        chunk_b = Chunk(chunk_id='b', heading='Refund Process',
                        content='Refunds are processed within 10 business days after the approval is confirmed.',
                        words=12, source_type='text')
        out = ConsistencyChecker(A32Config()).run([chunk_a, chunk_b])
        assert out.detected > 0
        assert out.findings[0].conflict_type == 'numeric'

    def test_confidence_scoring(self):
        from aiq.a32 import ConsistencyChecker, A32Config
        from aiq.core.types import Chunk
        chunk_a = Chunk(chunk_id='a', heading='Pricing',
                        content='Starter plan costs $19/month for up to 5 users with basic support.',
                        words=12, source_type='text')
        chunk_b = Chunk(chunk_id='b', heading='Pricing Update',
                        content='Starter plan costs $29/month for up to 5 users with email support.',
                        words=12, source_type='text')
        out = ConsistencyChecker(A32Config()).run([chunk_a, chunk_b])
        if out.findings:
            assert out.findings[0].confidence_score >= 0.0
            assert out.findings[0].confidence_score <= 1.0

    def test_no_contradiction_different_topics(self):
        from aiq.a32 import ConsistencyChecker, A32Config
        from aiq.core.types import Chunk
        chunk_a = Chunk(chunk_id='a', heading='Refund Policy',
                        content='Refunds take 5 business days to process after the request is approved.',
                        words=12, source_type='text')
        chunk_b = Chunk(chunk_id='b', heading='Free Trial',
                        content='The free trial period lasts 14 days from the date of initial signup.',
                        words=12, source_type='text')
        out = ConsistencyChecker(A32Config()).run([chunk_a, chunk_b])
        # Different topics should not be flagged as contradictions
        numeric_findings = [f for f in out.findings if f.conflict_type == 'numeric']
        assert len(numeric_findings) == 0
