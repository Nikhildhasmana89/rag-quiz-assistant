"""Unit tests for Neural Cross-Encoder Reranker."""

from unittest.mock import Mock, patch
import pytest

from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.rag_pipeline import RAGPipeline


class TestCrossEncoderReranker:
    """Test CrossEncoderReranker unit functionality."""

    def test_reranker_initialization(self):
        """Test reranker attributes and model name."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=16,
        )
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.batch_size == 16

    def test_reranker_empty_candidates(self):
        """Test reranking empty candidate list."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query="test query", candidates=[], top_k=5)
        assert results == []

    def test_reranker_empty_query(self):
        """Test reranking with empty or whitespace query returns candidates gracefully."""
        reranker = CrossEncoderReranker()
        candidates = [
            {"chunk_id": "c1", "text": "Document 1", "metadata": {"page": 1}, "hybrid_score": 0.8},
            {"chunk_id": "c2", "text": "Document 2", "metadata": {"page": 2}, "hybrid_score": 0.6},
        ]
        results = reranker.rerank(query="", candidates=candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["rerank_rank"] == 1
        assert results[1]["chunk_id"] == "c2"
        assert results[1]["rerank_rank"] == 2

    def test_reranker_single_candidate(self):
        """Test reranking a single candidate."""
        reranker = CrossEncoderReranker()
        mock_model = Mock()
        mock_model.predict.return_value = [2.45]
        reranker.model = mock_model

        candidate = {
            "chunk_id": "c1",
            "text": "Machine learning with PyTorch",
            "metadata": {"section": "AI"},
            "hybrid_score": 0.75,
        }
        results = reranker.rerank(
            query="deep learning", candidates=[candidate], top_k=5
        )
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["rerank_rank"] == 1
        assert results[0]["reranker_score"] == 2.45

    def test_reranker_scoring_and_sorting(self):
        """Test candidates are correctly sorted descending by reranker_score."""
        reranker = CrossEncoderReranker()
        mock_model = Mock()
        # Candidate 1 gets lower score than Candidate 2
        mock_model.predict.return_value = [-1.5, 4.8, 1.2]
        reranker.model = mock_model

        candidates = [
            {"chunk_id": "c1", "text": "Unrelated text", "metadata": {}, "hybrid_score": 0.9},
            {"chunk_id": "c2", "text": "Highly relevant answer", "metadata": {}, "hybrid_score": 0.5},
            {"chunk_id": "c3", "text": "Somewhat relevant", "metadata": {}, "hybrid_score": 0.6},
        ]

        results = reranker.rerank(
            query="specific question", candidates=candidates, top_k=3
        )

        assert len(results) == 3
        # Candidate 2 (score 4.8) should be rank 1
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["reranker_score"] == 4.8
        assert results[0]["rerank_rank"] == 1
        assert results[0]["initial_rank"] == 2

        # Candidate 3 (score 1.2) should be rank 2
        assert results[1]["chunk_id"] == "c3"
        assert results[1]["reranker_score"] == 1.2
        assert results[1]["rerank_rank"] == 2
        assert results[1]["initial_rank"] == 3

        # Candidate 1 (score -1.5) should be rank 3
        assert results[2]["chunk_id"] == "c1"
        assert results[2]["reranker_score"] == -1.5
        assert results[2]["rerank_rank"] == 3
        assert results[2]["initial_rank"] == 1

    def test_reranker_top_k_cutoff(self):
        """Test reranking respects top_k limit."""
        reranker = CrossEncoderReranker()
        mock_model = Mock()
        mock_model.predict.return_value = [1.0, 2.0, 3.0, 4.0, 5.0]
        reranker.model = mock_model

        candidates = [
            {"chunk_id": f"c{i}", "text": f"Text {i}", "metadata": {}}
            for i in range(5)
        ]
        results = reranker.rerank(query="test", candidates=candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c4"
        assert results[1]["chunk_id"] == "c3"

    def test_reranker_preserves_step2_metadata(self):
        """Test that all Step 2 metadata fields are strictly preserved."""
        reranker = CrossEncoderReranker()
        mock_model = Mock()
        mock_model.predict.return_value = [3.14]
        reranker.model = mock_model

        full_meta = {
            "document_id": "doc_abc123",
            "source_file": "paper.pdf",
            "page_number": 4,
            "section": "Methodology",
            "chunk_id": "doc_abc123_p4_c7",
            "chunk_index": 7,
            "chunk_size": 490,
            "document_type": "pdf",
        }
        candidate = {
            "chunk_id": "doc_abc123_p4_c7",
            "id": "doc_abc123_p4_c7",
            "text": "Detailed methodology description",
            "metadata": full_meta,
            "dense_score": 0.82,
            "bm25_score": 7.4,
            "hybrid_score": 0.78,
            "retrieval_source": "hybrid",
            "rank": 3,
        }

        results = reranker.rerank(query="methodology details", candidates=[candidate], top_k=1)
        assert len(results) == 1
        res = results[0]
        assert res["metadata"] == full_meta
        assert res["metadata"]["document_id"] == "doc_abc123"
        assert res["metadata"]["page_number"] == 4
        assert res["metadata"]["section"] == "Methodology"
        assert res["chunk_id"] == "doc_abc123_p4_c7"
        assert res["id"] == "doc_abc123_p4_c7"
        assert res["dense_score"] == 0.82
        assert res["bm25_score"] == 7.4
        assert res["hybrid_score"] == 0.78
        assert res["retrieval_source"] == "hybrid"
        assert res["initial_rank"] == 3
        assert res["reranker_score"] == 3.14
        assert res["rerank_rank"] == 1

    def test_reranker_inference_error_fallback(self):
        """Test fallback when model.predict raises an exception."""
        reranker = CrossEncoderReranker()
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Inference out of memory")
        reranker.model = mock_model

        candidates = [
            {"chunk_id": "c1", "text": "First", "metadata": {}, "hybrid_score": 0.9},
            {"chunk_id": "c2", "text": "Second", "metadata": {}, "hybrid_score": 0.8},
        ]
        results = reranker.rerank(query="query", candidates=candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[1]["chunk_id"] == "c2"


class TestRAGPipelineRerankingIntegration:
    """Test RAGPipeline interaction with Neural Reranking."""

    def test_pipeline_retrieve_with_rerank_toggle(self):
        """Test retrieve_with_metadata can toggle reranking on and off."""
        mock_pdf = Mock()
        mock_chunker = Mock()
        mock_store = Mock()
        mock_llm = Mock()

        # Mock hybrid retriever
        mock_hybrid = Mock()
        mock_hybrid.retrieve_with_metadata.return_value = [
            {"chunk_id": "c1", "text": "Text 1", "metadata": {}, "hybrid_score": 0.9, "rank": 1},
            {"chunk_id": "c2", "text": "Text 2", "metadata": {}, "hybrid_score": 0.7, "rank": 2},
        ]

        # Mock reranker
        mock_reranker = Mock()
        mock_reranker.model = Mock()
        mock_reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_reranker.rerank.return_value = [
            {"chunk_id": "c2", "text": "Text 2", "metadata": {}, "reranker_score": 5.0, "rerank_rank": 1, "rank": 1},
            {"chunk_id": "c1", "text": "Text 1", "metadata": {}, "reranker_score": 1.0, "rerank_rank": 2, "rank": 2},
        ]

        # Config with reranking enabled
        mock_config = Mock()
        mock_config.hybrid = Mock(retrieval_mode="hybrid", dense_weight=0.5, bm25_weight=0.5, fusion_method="rrf")
        mock_config.rerank = Mock(enabled=True, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", candidate_top_k=20, final_top_k=5)

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_store,
            llm_client=mock_llm,
            hybrid_retriever=mock_hybrid,
            reranker=mock_reranker,
            config=mock_config,
        )

        # 1. Retrieve with reranking enabled (default)
        res_reranked = pipeline.retrieve_with_metadata("test query", n_results=2, rerank=True)
        assert len(res_reranked) == 2
        assert res_reranked[0]["chunk_id"] == "c2"
        mock_reranker.rerank.assert_called_once()

        # 2. Retrieve with reranking disabled (rerank=False)
        mock_reranker.rerank.reset_mock()
        res_raw_hybrid = pipeline.retrieve_with_metadata("test query", n_results=2, rerank=False)
        assert len(res_raw_hybrid) == 2
        assert res_raw_hybrid[0]["chunk_id"] == "c1"
        mock_reranker.rerank.assert_not_called()

    def test_pipeline_status_includes_reranker_info(self):
        """Test get_status reflects reranker configuration."""
        mock_pdf = Mock()
        mock_pdf.method = "pdfplumber"
        mock_chunker = Mock(chunk_size=500, chunk_overlap=0)
        mock_store = Mock()
        mock_store.get_collection_info.return_value = {"document_count": 6}
        mock_llm = Mock()
        mock_llm.get_model_info.return_value = {"provider": "groq", "model": "llama3"}

        mock_reranker = Mock()
        mock_reranker.model = Mock()
        mock_reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        mock_config = Mock()
        mock_config.hybrid = Mock(retrieval_mode="hybrid", dense_weight=0.5, bm25_weight=0.5, fusion_method="weighted")
        mock_config.rerank = Mock(enabled=True, candidate_top_k=20, final_top_k=5)

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_store,
            llm_client=mock_llm,
            reranker=mock_reranker,
            config=mock_config,
        )

        status = pipeline.get_status()
        assert "reranker" in status
        assert status["reranker"]["enabled"] is True
        assert status["reranker"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert status["reranker"]["candidate_top_k"] == 20
        assert status["reranker"]["final_top_k"] == 5
