"""Unit tests for Hybrid Retrieval (BM25 + Dense Vector Search + Fusion)."""

import json
import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from app.retrieval.bm25_retriever import BM25Retriever, default_tokenizer
from app.retrieval.hybrid_retriever import HybridRetriever, min_max_normalize
from app.rag_pipeline import RAGPipeline


class TestBM25Retriever:
    """Test BM25 keyword retrieval, tokenization, and persistence."""

    def test_default_tokenizer(self):
        """Test default tokenization with lowercase and punctuation stripping."""
        tokens = default_tokenizer("Hello, World! This is a test: 123.")
        assert tokens == ["hello", "world", "this", "is", "a", "test", "123"]
        assert default_tokenizer("") == []
        assert default_tokenizer(None) == []

    def test_build_index_and_query(self):
        """Test indexing and basic keyword querying."""
        chunks = [
            {"chunk_id": "c1", "text": "Python machine learning models and neural networks", "metadata": {"page": 1}},
            {"chunk_id": "c2", "text": "PostgreSQL database optimization and SQL queries", "metadata": {"page": 2}},
            {"chunk_id": "c3", "text": "Frontend web development using React and CSS", "metadata": {"page": 3}},
        ]
        retriever = BM25Retriever()
        count = retriever.build_index(chunks)
        assert count == 3

        # Query relevant term
        results = retriever.query("machine learning", n_results=5)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["bm25_score"] > 0.0
        assert results[0]["bm25_rank"] == 1
        assert results[0]["metadata"]["page"] == 1

    def test_query_unknown_terms_and_empty(self):
        """Test query with terms not in corpus and empty query string."""
        chunks = [
            {"chunk_id": "c1", "text": "Python programming and algorithms", "metadata": {}},
        ]
        retriever = BM25Retriever()
        retriever.build_index(chunks)

        assert retriever.query("quantum thermodynamics") == []
        assert retriever.query("") == []
        assert retriever.query("   ") == []

    def test_persistence_save_and_load(self, tmp_path):
        """Test saving BM25 index to JSON and loading it back."""
        index_file = tmp_path / "bm25_index.json"
        chunks = [
            {"chunk_id": "c1", "text": "Alpha beta gamma", "metadata": {"section": "math"}},
            {"chunk_id": "c2", "text": "Delta epsilon zeta", "metadata": {"section": "physics"}},
        ]
        retriever = BM25Retriever(persist_path=index_file)
        retriever.build_index(chunks)
        retriever.save()

        assert index_file.exists()

        # Load into new retriever instance
        new_retriever = BM25Retriever(persist_path=index_file)
        loaded = new_retriever.load()
        assert loaded is True
        assert len(new_retriever.corpus_chunks) == 2

        results = new_retriever.query("epsilon")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    def test_add_documents_and_deduplication(self):
        """Test adding new documents updates existing chunks without duplication."""
        retriever = BM25Retriever()
        retriever.build_index([
            {"chunk_id": "c1", "text": "Initial text for chunk 1", "metadata": {}},
        ])

        # Add new chunk and update c1
        retriever.add_documents([
            {"chunk_id": "c1", "text": "Updated text with machine learning", "metadata": {"updated": True}},
            {"chunk_id": "c2", "text": "Second chunk about databases", "metadata": {}},
        ])

        assert len(retriever.corpus_chunks) == 2
        results = retriever.query("machine learning")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["metadata"].get("updated") is True

    def test_sync_from_vector_store(self):
        """Test synchronizing BM25 index from a vector store collection."""
        mock_store = Mock()
        mock_collection = Mock()
        mock_store.collection = mock_collection
        mock_collection.get.return_value = {
            "ids": ["doc1_c0", "doc1_c1"],
            "documents": ["First chunk text", "Second chunk text"],
            "metadatas": [{"page_number": 1}, {"page_number": 2}],
        }

        retriever = BM25Retriever()
        synced = retriever.sync_from_vector_store(mock_store)
        assert synced == 2
        assert len(retriever.corpus_chunks) == 2
        assert retriever.corpus_chunks[0]["chunk_id"] == "doc1_c0"
        assert retriever.corpus_chunks[1]["metadata"]["page_number"] == 2


class TestHybridRetriever:
    """Test HybridRetriever score normalization, fusion methods, and deduplication."""

    def test_min_max_normalize(self):
        """Test score normalization edge cases."""
        # Standard case
        scores = [10.0, 20.0, 30.0]
        norm = min_max_normalize(scores)
        assert norm == [0.0, 0.5, 1.0]

        # Identical positive scores
        assert min_max_normalize([5.0, 5.0]) == [1.0, 1.0]

        # Identical zero scores
        assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]

        # Empty list
        assert min_max_normalize([]) == []

    def test_weighted_score_fusion_and_deduplication(self):
        """Test weighted score fusion properly deduplicates and computes scores."""
        dense_results = [
            {"chunk_id": "c1", "text": "Chunk 1", "metadata": {"page": 1}, "dense_score": 0.9, "dense_rank": 1},
            {"chunk_id": "c2", "text": "Chunk 2", "metadata": {"page": 2}, "dense_score": 0.5, "dense_rank": 2},
        ]
        bm25_results = [
            {"chunk_id": "c2", "text": "Chunk 2", "metadata": {"page": 2}, "bm25_score": 12.0, "bm25_rank": 1},
            {"chunk_id": "c3", "text": "Chunk 3", "metadata": {"page": 3}, "bm25_score": 6.0, "bm25_rank": 2},
        ]

        retriever = HybridRetriever(
            vector_store=Mock(),
            bm25_retriever=Mock(),
            dense_weight=0.5,
            bm25_weight=0.5,
        )

        fused = retriever.fuse_weighted(dense_results, bm25_results, alpha=0.5)

        # 3 unique chunks
        assert len(fused) == 3
        chunk_ids = [c["chunk_id"] for c in fused]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids

        # c2 appeared in both -> source must be "hybrid"
        c2_item = next(c for c in fused if c["chunk_id"] == "c2")
        assert c2_item["retrieval_source"] == "hybrid"
        assert c2_item["dense_score"] == 0.5
        assert c2_item["bm25_score"] == 12.0

        # c1 only in dense, c3 only in bm25
        c1_item = next(c for c in fused if c["chunk_id"] == "c1")
        assert c1_item["retrieval_source"] == "dense"
        c3_item = next(c for c in fused if c["chunk_id"] == "c3")
        assert c3_item["retrieval_source"] == "bm25"

        # Ranks must be 1, 2, 3
        assert [c["rank"] for c in fused] == [1, 2, 3]

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion computes correct scores: 1/(k + rank)."""
        dense_results = [
            {"chunk_id": "c1", "text": "Chunk 1", "metadata": {}, "dense_score": 0.8},
            {"chunk_id": "c2", "text": "Chunk 2", "metadata": {}, "dense_score": 0.6},
        ]
        bm25_results = [
            {"chunk_id": "c2", "text": "Chunk 2", "metadata": {}, "bm25_score": 10.0},
            {"chunk_id": "c3", "text": "Chunk 3", "metadata": {}, "bm25_score": 5.0},
        ]

        retriever = HybridRetriever(
            vector_store=Mock(),
            bm25_retriever=Mock(),
            rrf_k=60,
        )

        fused = retriever.fuse_rrf(dense_results, bm25_results, k=60)
        assert len(fused) == 3

        # c2 rank in dense is 2, rank in bm25 is 1
        # c2 RRF score = 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61
        c2_item = next(c for c in fused if c["chunk_id"] == "c2")
        expected_c2_rrf = (1.0 / 62.0) + (1.0 / 61.0)
        assert abs(c2_item["hybrid_score"] - expected_c2_rrf) < 1e-6
        assert c2_item["retrieval_source"] == "hybrid"

    def test_fallback_when_one_pool_is_empty(self):
        """Test hybrid retrieval cleanly falls back when dense or BM25 returns nothing."""
        mock_dense = Mock()
        mock_bm25 = Mock()

        # Dense returns nothing, BM25 returns 1 item
        mock_dense.query.return_value = {"documents": [[]]}
        mock_bm25.query.return_value = [
            {"chunk_id": "c1", "text": "Only BM25 match", "metadata": {}, "bm25_score": 8.0}
        ]

        retriever = HybridRetriever(
            vector_store=mock_dense,
            bm25_retriever=mock_bm25,
        )

        results = retriever.retrieve_hybrid("keyword query", n_results=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["retrieval_source"] == "bm25"


class TestRAGPipelineHybridIntegration:
    """Test RAGPipeline interaction with Hybrid and Vector retrieval modes."""

    def test_pipeline_retrieve_modes(self):
        """Test retrieve with explicit mode='vector' and mode='hybrid'."""
        mock_pdf = Mock()
        mock_chunker = Mock()
        mock_store = Mock()
        mock_llm = Mock()

        mock_store.query.return_value = {
            "documents": [["Dense text chunk"]],
            "metadatas": [[{"page_number": 1, "chunk_id": "d1"}]],
            "ids": [["d1"]],
            "distances": [[0.1]],
        }

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_store,
            llm_client=mock_llm,
        )

        # Mode: vector
        docs_vec = pipeline.retrieve("test", n_results=1, mode="vector")
        assert len(docs_vec) == 1
        assert docs_vec[0] == "Dense text chunk"

        # retrieve_with_metadata: vector
        meta_vec = pipeline.retrieve_with_metadata("test", n_results=1, mode="vector")
        assert len(meta_vec) == 1
        assert meta_vec[0]["retrieval_source"] == "vector"
        assert meta_vec[0]["id"] == "d1"

    def test_pipeline_status_includes_hybrid(self):
        """Test get_status reflects hybrid retriever configuration."""
        mock_pdf = Mock()
        mock_pdf.method = "pdfplumber"
        mock_chunker = Mock()
        mock_chunker.chunk_size = 500
        mock_chunker.chunk_overlap = 0
        mock_store = Mock()
        mock_store.get_collection_info.return_value = {"document_count": 5}
        mock_llm = Mock()
        mock_llm.get_model_info.return_value = {"provider": "groq", "model": "llama3"}

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_store,
            llm_client=mock_llm,
        )

        status = pipeline.get_status()
        assert status["vector_store"] == "chromadb"
        assert "hybrid_retriever" in status
        assert status["hybrid_retriever"]["enabled"] is True
