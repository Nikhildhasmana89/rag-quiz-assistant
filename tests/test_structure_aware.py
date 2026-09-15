"""Unit tests for Step 2: Structure-Aware Document Processing and Metadata."""

import pytest
import hashlib
from unittest.mock import Mock
from app.text_chunker import TextChunker, detect_section_heading
from app.rag_pipeline import RAGPipeline


class TestStructureAwareChunking:
    """Test structure-aware chunking and metadata preservation."""

    def test_section_heading_detection_standard_headers(self):
        """Test detection of common academic and technical section headings."""
        test_cases = [
            ("Abstract", "Abstract"),
            ("1. Introduction", "Introduction"),
            ("2. Related Work", "Related Work"),
            ("3. Proposed Methodology", "Proposed Methodology"),
            ("4. System Architecture", "System Architecture"),
            ("5. Experiments and Results", "Experiments And Results"),
            ("6. Discussion", "Discussion"),
            ("7. Conclusion", "Conclusion"),
            ("8. References", "References"),
            ("Technical Skills", "Technical Skills"),
            ("Work Experience", "Work Experience"),
            ("Education", "Education"),
            ("Projects", "Projects"),
        ]
        for header_text, expected in test_cases:
            detected = detect_section_heading(header_text)
            assert detected is not None, f"Failed to detect: {header_text}"
            assert expected.lower() in detected.lower(), f"Expected {expected} in {detected}"

    def test_section_heading_detection_rejects_body_text(self):
        """Test that regular prose or very long lines are not classified as headings."""
        body_lines = [
            "This is a regular sentence describing the introduction of our method.",
            "We used Python and PyTorch for implementing the system architecture in our university lab.",
            "A" * 120,  # Line too long
            "",
            "   ",
        ]
        for line in body_lines:
            assert detect_section_heading(line) is None

    def test_chunk_page_with_metadata_retention(self):
        """Test that chunk_page_with_metadata retains page number, doc ID, and chunk ID."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        page_text = (
            "Introduction\n\n"
            "Retrieval Augmented Generation combines dense semantic search with neural generative language models. "
            "It helps reduce hallucination in complex enterprise domains.\n\n"
            "Methodology\n\n"
            "We divide documents into semantic chunks and index them in a high-dimensional vector space."
        )

        chunks = chunker.chunk_page_with_metadata(
            page_text=page_text,
            page_number=2,
            document_id="doc_test123",
            source_file="research_paper.pdf",
            document_type="pdf",
            start_chunk_index=0,
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["document_id"] == "doc_test123"
            assert chunk["source_file"] == "research_paper.pdf"
            assert chunk["page_number"] == 2
            assert chunk["document_type"] == "pdf"
            assert chunk["chunk_id"].startswith("doc_test123_p2_c")
            assert isinstance(chunk["chunk_index"], int)
            assert chunk["chunk_size"] == len(chunk["text"])
            assert "section" in chunk

        # Verify section mapping
        intro_chunks = [c for c in chunks if "Retrieval" in c["text"]]
        if intro_chunks:
            assert intro_chunks[0]["section"] == "Introduction"

        method_chunks = [c for c in chunks if "divide documents" in c["text"]]
        if method_chunks:
            assert method_chunks[0]["section"] == "Methodology"

    def test_deterministic_chunk_id_uniqueness(self):
        """Test that generated chunk IDs are globally unique across multiple pages."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        doc_id = "doc_unique_test"
        
        all_chunk_ids = []
        global_idx = 0
        for page_num in range(1, 4):
            page_text = f"This is page {page_num} content with enough text to generate multiple chunks. " * 3
            page_chunks = chunker.chunk_page_with_metadata(
                page_text=page_text,
                page_number=page_num,
                document_id=doc_id,
                source_file="multi_page.pdf",
                start_chunk_index=global_idx,
            )
            for c in page_chunks:
                all_chunk_ids.append(c["chunk_id"])
                global_idx += 1

        # Check total IDs match unique IDs (zero collisions)
        assert len(all_chunk_ids) == len(set(all_chunk_ids))
        assert len(all_chunk_ids) >= 3


class TestRAGPipelineStructureAwareness:
    """Test RAGPipeline structure awareness and metadata retrieval."""

    def test_retrieve_with_metadata_structure(self):
        """Test that retrieve_with_metadata returns full provenance dictionary."""
        mock_pdf_extractor = Mock()
        mock_text_chunker = Mock()
        mock_vector_store = Mock()
        mock_llm_client = Mock()

        mock_vector_store.collection = Mock()
        mock_vector_store.query.return_value = {
            "documents": [["Chunk 1 text", "Chunk 2 text"]],
            "metadatas": [[
                {
                    "document_id": "doc_abc123",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "section": "Introduction",
                    "chunk_id": "doc_abc123_p1_c0",
                },
                {
                    "document_id": "doc_abc123",
                    "source_file": "sample.pdf",
                    "page_number": 2,
                    "section": "Methodology",
                    "chunk_id": "doc_abc123_p2_c1",
                },
            ]],
            "ids": [["doc_abc123_p1_c0", "doc_abc123_p2_c1"]],
            "distances": [[0.12, 0.35]],
        }

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf_extractor,
            text_chunker=mock_text_chunker,
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        results = pipeline.retrieve_with_metadata("test query", n_results=2)
        assert len(results) == 2
        assert results[0]["text"] == "Chunk 1 text"
        assert results[0]["metadata"]["page_number"] == 1
        assert results[0]["metadata"]["section"] == "Introduction"
        assert results[0]["id"] == "doc_abc123_p1_c0"
        assert results[0]["distance"] == 0.12

    def test_backward_compatible_retrieve(self):
        """Test that original retrieve() still returns List[str]."""
        mock_pdf_extractor = Mock()
        mock_text_chunker = Mock()
        mock_vector_store = Mock()
        mock_llm_client = Mock()

        mock_vector_store.collection = Mock()
        mock_vector_store.query.return_value = {
            "documents": [["Chunk 1 text", "Chunk 2 text"]],
        }

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf_extractor,
            text_chunker=mock_text_chunker,
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        results = pipeline.retrieve("test query")
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(results[0], str)
        assert results[0] == "Chunk 1 text"

    def test_rag_query_includes_both_documents_and_chunks(self):
        """Test that rag_query returns retrieved_documents and retrieved_chunks."""
        mock_pdf_extractor = Mock()
        mock_text_chunker = Mock()
        mock_vector_store = Mock()
        mock_llm_client = Mock()

        mock_vector_store.collection = Mock()
        mock_vector_store.query.return_value = {
            "documents": [["Chunk text"]],
            "metadatas": [[{"document_id": "doc_1", "page_number": 1, "section": "Summary"}]],
            "ids": [["doc_1_p1_c0"]],
            "distances": [[0.1]],
        }
        mock_llm_client.generate.return_value = "Answer based on context"

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf_extractor,
            text_chunker=mock_text_chunker,
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        result = pipeline.rag_query("What is this?")
        assert "retrieved_documents" in result
        assert "retrieved_chunks" in result
        assert result["retrieved_documents"] == ["Chunk text"]
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["metadata"]["page_number"] == 1
        assert result["response"] == "Answer based on context"
