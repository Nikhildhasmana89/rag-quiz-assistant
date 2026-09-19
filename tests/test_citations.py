"""Unit and regression tests for CitationEngine and Grounded Provenance (Step 6)."""

from unittest.mock import MagicMock
import pytest

from app.citations import (
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_SUPPORTED,
    CitationEngine,
)


@pytest.fixture
def sample_evidence_chunks():
    return [
        {
            "id": "chunk_001",
            "chunk_id": "chunk_001",
            "text": (
                "Nikhil Dhasmana worked at Stealth AI Startup from January 2023 to June 2024. "
                "He developed automated RAG pipelines and improved retrieval latency by 45%."
            ),
            "metadata": {
                "document_id": "doc_dda75b4545a4",
                "source_file": "Nikhil_Dhasmana_Resume_2.pdf",
                "page_number": 1,
                "section": "Professional Experience",
                "chunk_id": "chunk_001",
            },
            "retrieval_source": "hybrid",
            "reranker_score": 0.9421,
            "hybrid_score": 0.88,
        },
        {
            "id": "chunk_002",
            "chunk_id": "chunk_002",
            "text": (
                "Education: Bachelor of Technology in Computer Science and Engineering. "
                "Achieved cumulative GPA of 8.9 out of 10. Graduated with First Class Distinction."
            ),
            "metadata": {
                "document_id": "doc_dda75b4545a4",
                "source_file": "Nikhil_Dhasmana_Resume_2.pdf",
                "page_number": 2,
                "section": "Education",
                "chunk_id": "chunk_002",
            },
            "retrieval_source": "vector",
            "reranker_score": 0.7650,
            "hybrid_score": 0.65,
        },
    ]


class TestCitationEngine:
    """Tests for the grounded CitationEngine component."""

    def test_supported_claim_generates_citation(self, sample_evidence_chunks):
        engine = CitationEngine()
        claims = [
            {
                "claim": "Nikhil worked at Stealth AI Startup and improved latency by 45%.",
                "status": STATUS_SUPPORTED,
                "evidence_ids": ["chunk_001"],
            }
        ]
        answer = "Nikhil worked at Stealth AI Startup and improved latency by 45%."

        result = engine.generate_citations(claims, sample_evidence_chunks, answer)

        assert len(result["citations"]) == 1
        cit = result["citations"][0]
        assert cit["citation_id"] == "cit_1"
        assert cit["citation_index"] == 1
        assert cit["claim_status"] == STATUS_SUPPORTED
        assert cit["chunk_id"] == "chunk_001"
        assert cit["document_name"] == "Nikhil_Dhasmana_Resume_2.pdf"
        assert cit["document_id"] == "doc_dda75b4545a4"
        assert cit["page_number"] == 1
        assert cit["section"] == "Professional Experience"
        assert cit["reranker_score"] == 0.9421
        assert "improved retrieval latency by 45%" in cit["evidence_text"]
        assert len(result["unsupported_claims"]) == 0

    def test_partially_supported_claim_generates_citation_with_notes(self, sample_evidence_chunks):
        engine = CitationEngine()
        claims = [
            {
                "claim": "Nikhil Dhasmana holds a B.Tech in CSE and also completed an MBA.",
                "status": STATUS_PARTIALLY_SUPPORTED,
                "evidence_ids": ["chunk_002"],
            }
        ]
        answer = "Nikhil Dhasmana holds a B.Tech in CSE and also completed an MBA."

        result = engine.generate_citations(claims, sample_evidence_chunks, answer)

        assert len(result["citations"]) == 1
        cit = result["citations"][0]
        assert cit["claim_status"] == STATUS_PARTIALLY_SUPPORTED
        assert cit["chunk_id"] == "chunk_002"
        assert cit["page_number"] == 2
        assert "Bachelor of Technology" in cit["evidence_text"]
        assert "Partially supported" in cit["notes"]

    def test_contradicted_claim_cites_conflicting_chunk(self, sample_evidence_chunks):
        engine = CitationEngine()
        claims = [
            {
                "claim": "Nikhil Dhasmana attended medical school and graduated in 2018.",
                "status": STATUS_CONTRADICTED,
                "evidence_ids": ["chunk_002"],
            }
        ]
        answer = "Nikhil Dhasmana attended medical school and graduated in 2018."

        result = engine.generate_citations(claims, sample_evidence_chunks, answer)

        assert len(result["citations"]) == 1
        cit = result["citations"][0]
        assert cit["claim_status"] == STATUS_CONTRADICTED
        assert cit["chunk_id"] == "chunk_002"
        assert "Contradicted" in cit["notes"]

    def test_insufficient_evidence_strictly_zero_citations(self, sample_evidence_chunks):
        engine = CitationEngine()
        claims = [
            {
                "claim": "The candidate has published five papers on quantum cryptography.",
                "status": STATUS_INSUFFICIENT_EVIDENCE,
                "evidence_ids": [],
            }
        ]
        answer = "The candidate has published five papers on quantum cryptography."

        result = engine.generate_citations(claims, sample_evidence_chunks, answer)

        # STRICT REQUIREMENT: Case 4 claims must produce ZERO citations!
        assert len(result["citations"]) == 0
        assert len(result["unsupported_claims"]) == 1
        assert result["unsupported_claims"][0] == claims[0]["claim"]
        assert result["claims"][0]["citation_ids"] == []

    def test_extract_evidence_snippet_verbatim(self, sample_evidence_chunks):
        engine = CitationEngine()
        chunk_text = sample_evidence_chunks[0]["text"]
        claim = "improved latency by 45%"

        snippet = engine.extract_evidence_snippet(chunk_text, claim)

        # Must be a 100% verbatim substring of the chunk
        assert snippet in chunk_text
        assert "45%" in snippet

    def test_annotated_response_markers(self, sample_evidence_chunks):
        engine = CitationEngine()
        claims = [
            {
                "claim": "Nikhil worked at Stealth AI Startup.",
                "status": STATUS_SUPPORTED,
                "evidence_ids": ["chunk_001"],
            }
        ]
        answer = "Nikhil worked at Stealth AI Startup. He enjoys reading in his free time."

        result = engine.generate_citations(claims, sample_evidence_chunks, answer)

        assert "[1]" in result["annotated_response"]
        # The unsupported sentence must not have a citation marker
        assert "free time" in result["annotated_response"]

    def test_format_citation_string(self):
        engine = CitationEngine()
        citation = {
            "document_name": "research_paper.pdf",
            "page_number": 7,
            "section": "Results",
            "chunk_id": "chunk_014",
            "evidence_text": "Under fine-tuning, the proposed model achieved 94.2% accuracy.",
        }

        formatted = engine.format_citation_string(citation)

        assert "[Document: research_paper.pdf | Page: 7 | Section: Results | Chunk: chunk_014]" in formatted
        assert 'Evidence: "Under fine-tuning, the proposed model achieved 94.2% accuracy."' in formatted

    def test_edge_cases_empty_inputs(self):
        engine = CitationEngine()

        # Empty claims
        res1 = engine.generate_citations([], [])
        assert res1["citations"] == []
        assert res1["annotated_response"] == ""

        # Empty evidence chunks
        claims = [{"claim": "Something", "status": STATUS_SUPPORTED, "evidence_ids": ["c1"]}]
        res2 = engine.generate_citations(claims, [])
        assert res2["citations"] == []
        assert len(res2["unsupported_claims"]) == 1


class TestPipelineCitationIntegration:
    """Tests verifying pipeline-level citation wiring."""

    def test_rag_query_with_citations(self):
        from app.rag_pipeline import RAGPipeline

        mock_pdf = MagicMock()
        mock_chunker = MagicMock()
        mock_vector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Nikhil Dhasmana has experience in RAG systems."

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = {
            "status": "supported",
            "claims": [
                {
                    "claim": "Nikhil Dhasmana has experience in RAG systems.",
                    "status": "supported",
                    "evidence_ids": ["chunk_test_1"],
                }
            ],
            "verification_latency_ms": 15.0,
        }

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_vector,
            llm_client=mock_llm,
            verifier=mock_verifier,
        )

        test_chunks = [
            {
                "chunk_id": "chunk_test_1",
                "id": "chunk_test_1",
                "text": "Nikhil Dhasmana has experience in RAG systems and LLM workflows.",
                "metadata": {
                    "document_id": "doc_test123",
                    "source_file": "Resume.pdf",
                    "page_number": 1,
                    "section": "Summary",
                },
                "retrieval_source": "vector",
                "reranker_score": 0.95,
            }
        ]
        pipeline.retrieve_with_metadata = MagicMock(return_value=test_chunks)

        result = pipeline.rag_query("What is Nikhil's experience?", verify=True, cite=True)

        assert result["citations_enabled"] is True
        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "chunk_test_1"
        assert result["citations"][0]["document_id"] == "doc_test123"
        assert result["citations"][0]["reranker_score"] == 0.95
        assert "[1]" in result["annotated_response"]
        assert "citation_latency_ms" in result
        assert result["total_latency_ms"] >= result["citation_latency_ms"]

    def test_rag_query_no_cite_flag(self):
        from app.rag_pipeline import RAGPipeline

        mock_pdf = MagicMock()
        mock_chunker = MagicMock()
        mock_vector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Some answer."

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_vector,
            llm_client=mock_llm,
        )
        pipeline.retrieve_with_metadata = MagicMock(return_value=[])

        result = pipeline.rag_query("Test query", cite=False)
        assert result["citations_enabled"] is False
        assert result["citations"] == []
