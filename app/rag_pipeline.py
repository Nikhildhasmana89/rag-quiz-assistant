"""RAG pipeline orchestrating the complete workflow with Hybrid Retrieval and Neural Reranking."""

import copy
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete Retrieval Augmented Generation pipeline with Hybrid Retrieval & Neural Reranking."""

    def __init__(
        self,
        pdf_extractor,
        text_chunker,
        vector_store,
        llm_client,
        hybrid_retriever=None,
        reranker=None,
        verifier=None,
        citation_engine=None,
        config=None,
    ):
        """Initialize RAG pipeline.

        Args:
            pdf_extractor: PDFTextExtractor instance
            text_chunker: TextChunker instance
            vector_store: ChromaVectorStore instance
            llm_client: LLMClient instance
            hybrid_retriever: Optional HybridRetriever instance
            reranker: Optional CrossEncoderReranker instance
            verifier: Optional EvidenceVerifier instance
            citation_engine: Optional CitationEngine instance
            config: Optional AppConfig instance
        """
        self.pdf_extractor = pdf_extractor
        self.text_chunker = text_chunker
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config

        # Initialize or wire hybrid retriever
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        elif config is not None:
            try:
                from app.retrieval import BM25Retriever, HybridRetriever

                persist_dir = "./data/bm25"
                if hasattr(config, "hybrid") and config.hybrid:
                    persist_dir = config.hybrid.bm25_persist_dir
                    dense_w = config.hybrid.dense_weight
                    bm25_w = config.hybrid.bm25_weight
                    fusion = config.hybrid.fusion_method
                    rrf_k = config.hybrid.rrf_k
                    dense_top_k = config.hybrid.dense_top_k
                    bm25_top_k = config.hybrid.bm25_top_k
                    final_top_k = config.hybrid.final_top_k
                else:
                    dense_w = 0.5
                    bm25_w = 0.5
                    fusion = "weighted"
                    rrf_k = 60
                    dense_top_k = 20
                    bm25_top_k = 20
                    final_top_k = 5

                bm25_file = Path(persist_dir) / "bm25_index.json"
                bm25 = BM25Retriever(persist_path=bm25_file)
                bm25.load()

                self.hybrid_retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    bm25_retriever=bm25,
                    dense_top_k=dense_top_k,
                    bm25_top_k=bm25_top_k,
                    final_top_k=final_top_k,
                    dense_weight=dense_w,
                    bm25_weight=bm25_w,
                    fusion_method=fusion,
                    rrf_k=rrf_k,
                )
            except Exception as e:
                logger.warning(f"Could not auto-initialize HybridRetriever: {e}")
                self.hybrid_retriever = None
        else:
            try:
                from app.retrieval import BM25Retriever, HybridRetriever

                # In-memory BM25 without loading from disk, preserving mock test isolation
                bm25 = BM25Retriever()
                self.hybrid_retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    bm25_retriever=bm25,
                )
            except Exception as e:
                logger.warning(f"Could not initialize default HybridRetriever: {e}")
                self.hybrid_retriever = None

        # Initialize or wire CrossEncoder reranker
        if reranker is not None:
            self.reranker = reranker
        elif config is not None and getattr(config, "rerank", None) and config.rerank.enabled:
            try:
                from app.reranking import CrossEncoderReranker

                self.reranker = CrossEncoderReranker(
                    model_name=config.rerank.model_name,
                    batch_size=config.rerank.batch_size,
                )
            except Exception as e:
                logger.warning(f"Could not auto-initialize CrossEncoderReranker: {e}")
                self.reranker = None
        else:
            self.reranker = None

        # Initialize or wire EvidenceVerifier

        if verifier is not None:
            self.verifier = verifier
        elif config is not None and getattr(config, "verification", None) and config.verification.enabled:
            try:
                from app.verification import EvidenceVerifier

                self.verifier = EvidenceVerifier(
                    llm_client=self.llm_client,
                    config=self.config,
                )
            except Exception as e:
                logger.warning(f"Could not auto-initialize EvidenceVerifier: {e}")
                self.verifier = None
        else:
            self.verifier = None

        # Initialize or wire CitationEngine (Step 6)
        if citation_engine is not None:
            self.citation_engine = citation_engine
        else:
            try:
                from app.citations import CitationEngine

                self.citation_engine = CitationEngine(config=self.config)
            except Exception as e:
                logger.warning(f"Could not auto-initialize CitationEngine: {e}")
                self.citation_engine = None

        if config and hasattr(config, "hybrid") and config.hybrid:
            self.retrieval_mode = config.hybrid.retrieval_mode
        else:
            self.retrieval_mode = "hybrid" if self.hybrid_retriever is not None else "vector"

        logger.info(
            f"Initialized RAGPipeline (retrieval_mode={self.retrieval_mode}, "
            f"reranker={'enabled' if (self.reranker and getattr(self.reranker, 'model', None)) else 'disabled'}, "
            f"verifier={'enabled' if self.verifier is not None else 'disabled'}, "
            f"citations={'enabled' if self.citation_engine is not None else 'disabled'})"
        )


    def ingest_pdf(
        self,
        pdf_path: Path,
        collection_name: Optional[str] = None,
        force_recreate: bool = False,
    ) -> Dict[str, Any]:
        """Ingest a PDF document into the vector store with structure awareness and provenance.

        Args:
            pdf_path: Path to PDF file
            collection_name: Optional collection name
            force_recreate: If True, recreate collection

        Returns:
            Ingestion results with metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Starting PDF ingestion: {pdf_path}")

        # Compute deterministic document ID based on content
        try:
            with open(pdf_path, "rb") as f:
                content_bytes = f.read()
            document_id = f"doc_{hashlib.sha256(content_bytes).hexdigest()[:12]}"
        except Exception:
            document_id = f"doc_{hashlib.sha256(str(pdf_path.name).encode()).hexdigest()[:12]}"

        # Extract text
        logger.debug("Extracting text from PDF...")
        texts = self.pdf_extractor.extract_text(pdf_path)
        logger.info(f"Extracted {len(texts)} pages")

        # Create or recreate collection
        if force_recreate:
            self.vector_store.delete_collection()
            if self.hybrid_retriever and getattr(self.hybrid_retriever, "bm25_retriever", None):
                self.hybrid_retriever.bm25_retriever.clear_index()

        self.vector_store.create_collection()

        # Chunk text with structure awareness and page metadata
        logger.debug("Chunking text with structure awareness...")
        all_chunks = []
        all_metadatas = []
        all_ids = []
        current_section = ""
        global_chunk_idx = 0

        # Try structure-aware page chunking if available
        if hasattr(self.text_chunker, "chunk_page_with_metadata") and callable(
            getattr(self.text_chunker, "chunk_page_with_metadata")
        ):
            try:
                for page_idx, page_text in enumerate(texts):
                    page_number = page_idx + 1
                    if not page_text or not str(page_text).strip():
                        continue
                    page_chunks = self.text_chunker.chunk_page_with_metadata(
                        page_text=str(page_text),
                        page_number=page_number,
                        document_id=document_id,
                        source_file=pdf_path.name,
                        document_type="pdf",
                        start_chunk_index=global_chunk_idx,
                        initial_section=current_section,
                    )
                    if isinstance(page_chunks, list):
                        for chunk_item in page_chunks:
                            if isinstance(chunk_item, dict) and "text" in chunk_item:
                                all_chunks.append(chunk_item["text"])
                                all_metadatas.append(chunk_item)
                                all_ids.append(
                                    chunk_item.get(
                                        "chunk_id",
                                        f"{document_id}_p{page_number}_c{global_chunk_idx}",
                                    )
                                )
                                if chunk_item.get("section"):
                                    current_section = chunk_item["section"]
                                global_chunk_idx += 1
            except Exception as e:
                logger.warning(f"Error during chunk_page_with_metadata: {e}")

        # Fallback to chunk_texts if page chunking didn't produce chunks (e.g. in mocked tests)
        if not all_chunks and hasattr(self.text_chunker, "chunk_texts"):
            fallback_chunks = self.text_chunker.chunk_texts(texts)
            if isinstance(fallback_chunks, list):
                for i, chunk in enumerate(fallback_chunks):
                    cid = f"{document_id}_p1_c{i}"
                    all_chunks.append(chunk)
                    all_ids.append(cid)
                    all_metadatas.append({
                        "text": chunk,
                        "document_id": document_id,
                        "source": str(pdf_path),
                        "source_file": pdf_path.name,
                        "page_number": 1,
                        "section": "",
                        "chunk_id": cid,
                        "chunk_index": i,
                        "chunk_size": len(chunk) if isinstance(chunk, str) else 0,
                        "document_type": "pdf",
                    })

        logger.info(f"Created {len(all_chunks)} chunks with metadata")

        # Add to vector store
        logger.debug("Adding chunks to vector store...")
        self.vector_store.add_documents(
            texts=all_chunks,
            ids=all_ids,
            metadata=all_metadatas,
        )

        # Add to BM25 index and persist
        if self.hybrid_retriever and getattr(self.hybrid_retriever, "bm25_retriever", None):
            bm25_chunks = [
                {"chunk_id": cid, "text": txt, "metadata": meta}
                for cid, txt, meta in zip(all_ids, all_chunks, all_metadatas)
            ]
            self.hybrid_retriever.bm25_retriever.add_documents(bm25_chunks)
            if self.hybrid_retriever.bm25_retriever.persist_path:
                try:
                    self.hybrid_retriever.bm25_retriever.save()
                except Exception as e:
                    logger.debug(f"Could not persist BM25 index: {e}")

        result = {
            "success": True,
            "document_id": document_id,
            "pdf_path": str(pdf_path),
            "pages_extracted": len(texts),
            "chunks_created": len(all_chunks),
            "collection_name": self.vector_store.collection_name,
        }

        logger.info(f"Successfully ingested PDF: {result}")
        return result

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            n_results: Number of results to retrieve
            mode: Optional 'vector' or 'hybrid' (defaults to self.retrieval_mode)
            rerank: Optional boolean to enable/disable reranking

        Returns:
            List of relevant document chunk strings
        """
        results = self.retrieve_with_metadata(
            query=query,
            n_results=n_results,
            mode=mode,
            rerank=rerank,
        )
        return [c["text"] for c in results if "text" in c]

    def retrieve_with_metadata(
        self,
        query: str,
        n_results: int = 5,
        mode: Optional[str] = None,
        fusion_method: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents along with provenance, retrieval scores, and optional reranker scores.

        Args:
            query: Query text
            n_results: Number of results to retrieve
            mode: Optional 'vector' or 'hybrid' (defaults to self.retrieval_mode)
            fusion_method: Optional 'weighted' or 'rrf'
            rerank: Optional boolean to toggle neural reranking (defaults to config setting)

        Returns:
            List of candidate dictionaries
        """
        logger.debug(f"Retrieving documents with metadata for query: {query}")
        active_mode = (mode or self.retrieval_mode).lower()

        # Pure vector retrieval mode
        if active_mode == "vector" or not self.hybrid_retriever:
            if hasattr(self.vector_store, "collection") and self.vector_store.collection is None:
                self.vector_store.create_collection()

            results = self.vector_store.query(
                query_texts=[query],
                n_results=n_results,
            )

            if not results or "documents" not in results or not results["documents"]:
                return []

            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            structured = []
            for rank, (doc, meta, cid, dist) in enumerate(zip(docs, metas, ids, dists), start=1):
                dist_val = float(dist) if dist is not None else 0.0
                similarity = 1.0 / (1.0 + max(0.0, dist_val))
                chunk_id = cid or (meta.get("chunk_id") if isinstance(meta, dict) else "")
                structured.append({
                    "text": doc,
                    "metadata": meta or {},
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "distance": dist_val,
                    "dense_score": similarity,
                    "hybrid_score": similarity,
                    "retrieval_source": "vector",
                    "rank": rank,
                })

            logger.info(f"Retrieved {len(structured)} documents with metadata (vector)")
            return structured

        # Hybrid retrieval mode: determine whether to rerank
        should_rerank = rerank
        if should_rerank is None:
            if self.reranker is not None and getattr(self.reranker, "model", None) is not None:
                should_rerank = (
                    self.config.rerank.enabled
                    if (self.config and getattr(self.config, "rerank", None))
                    else True
                )
            else:
                should_rerank = False

        if should_rerank and self.reranker is not None and getattr(self.reranker, "model", None) is not None:
            # Retrieve candidate pool of size candidate_top_k
            candidate_k = 20
            if self.config and getattr(self.config, "rerank", None):
                candidate_k = max(n_results, self.config.rerank.candidate_top_k)
            else:
                candidate_k = max(n_results, 20)

            candidate_pool = self.hybrid_retriever.retrieve_with_metadata(
                query=query,
                n_results=candidate_k,
                mode="hybrid",
                fusion_method=fusion_method,
            )
            for c in candidate_pool:
                if "id" not in c:
                    c["id"] = c.get("chunk_id", "")
                if "distance" not in c:
                    c["distance"] = c.get("distance", 0.0)

            reranked = self.reranker.rerank(
                query=query,
                candidates=candidate_pool,
                top_k=n_results,
            )
            logger.info(f"Retrieved and reranked {len(reranked)} documents (hybrid+rerank)")
            return reranked

        # Hybrid retrieval without reranking
        candidates = self.hybrid_retriever.retrieve_with_metadata(
            query=query,
            n_results=n_results,
            mode="hybrid",
            fusion_method=fusion_method,
        )
        for c in candidates:
            if "id" not in c:
                c["id"] = c.get("chunk_id", "")
            if "distance" not in c:
                c["distance"] = c.get("distance", 0.0)

        logger.info(f"Retrieved {len(candidates)} documents with metadata (hybrid)")
        return candidates

    def generate_response(
        self,
        query: str,
        retrieved_documents: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate LLM response based on retrieved documents.

        Args:
            query: Original query
            retrieved_documents: Documents retrieved from vector store
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated response
        """
        logger.debug("Generating LLM response...")

        # Create context
        context = "\n\n".join(retrieved_documents)

        # Create prompt
        prompt = f"""You are a helpful expert assistant. Your users are asking questions based on provided documents.
You will be shown the user's question and relevant information from the documents.
Answer the user's question using only the provided information.
If the information is not sufficient to answer the question, say so clearly.

Question: {query}

Information from documents:
{context}

Answer:"""

        logger.debug(f"Generated prompt ({len(prompt)} chars)")

        # Generate response
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info("Successfully generated response")
        return response

    def rag_query(
        self,
        query: str,
        n_retrieve: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        mode: Optional[str] = None,
        fusion_method: Optional[str] = None,
        rerank: Optional[bool] = None,
        verify: Optional[bool] = None,
        cite: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute complete RAG query: retrieve (+ optional rerank) + generate (+ optional verify) + citations.

        Args:
            query: Query text
            n_retrieve: Number of documents to retrieve
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            mode: Optional 'vector' or 'hybrid'
            fusion_method: Optional 'weighted' or 'rrf'
            rerank: Optional boolean to toggle reranking
            verify: Optional boolean to toggle evidence verification
            cite: Optional boolean to toggle citation generation

        Returns:
            Result dict with query, retrieved docs, metadata, response, verification, citations, and latency info
        """
        logger.info(f"Executing RAG query: {query}")
        active_mode = (mode or self.retrieval_mode).lower()

        t0_retrieval = time.perf_counter()
        retrieved_chunks = self.retrieve_with_metadata(
            query=query,
            n_results=n_retrieve,
            mode=active_mode,
            fusion_method=fusion_method,
            rerank=rerank,
        )
        retrieval_latency_ms = (time.perf_counter() - t0_retrieval) * 1000.0
        documents = [c["text"] for c in retrieved_chunks] if retrieved_chunks else []

        # Generate response
        t0_gen = time.perf_counter()
        response = self.generate_response(
            query=query,
            retrieved_documents=documents,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generation_latency_ms = (time.perf_counter() - t0_gen) * 1000.0

        # Verification Stage (Step 5)
        should_verify = verify
        if should_verify is None:
            if self.verifier is not None:
                should_verify = (
                    self.config.verification.enabled
                    if (self.config and getattr(self.config, "verification", None))
                    else True
                )
            else:
                should_verify = False

        verification_result = None
        if should_verify and self.verifier is not None:
            try:
                verification_result = self.verifier.verify(
                    query=query,
                    answer=response,
                    evidence=retrieved_chunks,
                )
            except Exception as e:
                logger.warning(f"Verification execution failed: {e}")
                verification_result = {
                    "status": "insufficient_evidence",
                    "supported_claims": [],
                    "unsupported_claims": [response],
                    "contradicted_claims": [],
                    "claims": [{"claim": response, "status": "insufficient_evidence", "evidence_ids": []}],
                    "evidence": [],
                    "verification_latency_ms": 0.0,
                    "error": str(e),
                }

        # Grounded Citation Generation Stage (Step 6)
        should_cite = cite
        if should_cite is None:
            should_cite = True if self.citation_engine is not None else False

        citation_result = None
        citation_latency_ms = 0.0
        if should_cite and self.citation_engine is not None:
            try:
                if verification_result is not None and "claims" in verification_result and verification_result["claims"]:
                    eval_claims = verification_result["claims"]
                else:
                    eval_claims = [{
                        "claim": response,
                        "status": "supported" if retrieved_chunks else "insufficient_evidence",
                        "evidence_ids": [c.get("chunk_id") or c.get("id") for c in retrieved_chunks[:3]],
                    }]

                citation_result = self.citation_engine.generate_citations(
                    claims=eval_claims,
                    evidence_chunks=retrieved_chunks,
                    answer_text=response,
                )
                citation_latency_ms = citation_result.get("citation_latency_ms", 0.0)
            except Exception as e:
                logger.warning(f"Citation generation failed: {e}")
                citation_result = {
                    "citations": [],
                    "claims": [],
                    "annotated_response": response,
                    "unsupported_claims": [],
                    "citation_latency_ms": 0.0,
                    "error": str(e),
                }

        result = {
            "query": query,
            "retrieved_documents": documents,
            "retrieved_chunks": retrieved_chunks,
            "response": response,
            "n_documents_retrieved": len(documents),
            "retrieval_mode": active_mode,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
        }
        if active_mode == "hybrid" and self.hybrid_retriever:
            result["fusion_method"] = fusion_method or getattr(
                self.hybrid_retriever, "fusion_method", "weighted"
            )
            is_reranked = (
                (rerank is not False)
                and (self.reranker is not None)
                and (getattr(self.reranker, "model", None) is not None)
            )
            result["reranking_enabled"] = is_reranked
            if is_reranked:
                result["reranker_model"] = self.reranker.model_name

        if verification_result is not None:
            result["verification"] = verification_result
            result["verification_enabled"] = True
            result["verification_latency_ms"] = verification_result.get("verification_latency_ms", 0.0)
        else:
            result["verification_enabled"] = False
            result["verification_latency_ms"] = 0.0

        if citation_result is not None:
            result["citations"] = citation_result.get("citations", [])
            result["claims"] = citation_result.get("claims", [])
            result["annotated_response"] = citation_result.get("annotated_response", response)
            result["citations_enabled"] = True
            result["citation_latency_ms"] = citation_latency_ms
            if "unsupported_claims" in citation_result:
                result["unsupported_claims"] = citation_result["unsupported_claims"]
            if "metrics" in citation_result:
                result["citation_metrics"] = citation_result["metrics"]
        else:
            result["citations"] = []
            result["claims"] = []
            result["annotated_response"] = response
            result["citations_enabled"] = False
            result["citation_latency_ms"] = 0.0

        total_lat = (
            retrieval_latency_ms
            + generation_latency_ms
            + result["verification_latency_ms"]
            + result["citation_latency_ms"]
        )
        result["total_latency_ms"] = round(total_lat, 2)

        logger.info("RAG query completed successfully")
        return result


    def generate_quiz(
        self,
        num_questions: int = 5,
        n_context_chunks: int = 8,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Generate interview-style quiz questions grounded in the indexed document.

        Args:
            num_questions: How many questions to generate
            n_context_chunks: How many indexed chunks to draw the questions from
            temperature: Sampling temperature for generation

        Returns:
            List of question dicts with id, question, expected_answer, context
        """
        from app.utils import extract_json

        logger.info(f"Generating {num_questions} quiz questions from indexed content")

        chunks = self.vector_store.sample_documents(n=n_context_chunks)
        if not chunks:
            raise ValueError("No indexed content found. Ingest a PDF before generating a quiz.")

        context = "\n\n---\n\n".join(chunks)

        prompt = f"""You are an experienced technical interviewer preparing candidate practice questions.
Using ONLY the study material below, write {num_questions} interview-style questions that test understanding of the material.

For each question, also provide the key points a strong answer should include, based strictly on the material.

Study material:
{context}

Respond with ONLY a valid JSON array, no other text, in exactly this format:
[
  {{
    "question": "the interview question",
    "expected_answer": "2-4 key points a good answer should cover, based on the material"
  }}
]"""

        raw = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=1500,
        )

        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            raise ValueError("Quiz generation did not return a list of questions")

        questions = []
        for i, item in enumerate(parsed[:num_questions]):
            questions.append({
                "id": f"q{i + 1}",
                "question": item.get("question", "").strip(),
                "expected_answer": item.get("expected_answer", "").strip(),
                "context": context,
            })

        logger.info(f"Generated {len(questions)} quiz questions")
        return questions

    def evaluate_answer(
        self,
        question: str,
        expected_answer: str,
        context: str,
        user_answer: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Grade a user's answer to a generated quiz question.

        Args:
            question: The quiz question
            expected_answer: Key points a good answer should include
            context: The source material the question was generated from
            user_answer: The candidate's typed answer
            temperature: Sampling temperature (kept low for consistent grading)

        Returns:
            Dict with: verdict, score (0-100), feedback
        """
        from app.utils import extract_json

        prompt = f"""You are grading a candidate's interview answer.

Question: {question}

Reference material (ground truth): {context}

Key points a strong answer should cover: {expected_answer}

Candidate's answer: {user_answer}

Grade the answer strictly based on the reference material and key points above.
Respond with ONLY valid JSON, no other text, in exactly this format:
{{
  "verdict": "Correct" or "Partially Correct" or "Incorrect",
  "score": <integer 0-100>,
  "feedback": "2-3 sentences: what was right, what was missing or wrong"
}}"""

        raw = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=500,
        )

        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Answer evaluation did not return a JSON object")

        return {
            "verdict": parsed.get("verdict", "Unknown"),
            "score": parsed.get("score", 0),
            "feedback": parsed.get("feedback", ""),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and configuration.

        Returns:
            Status information
        """
        try:
            collection_info = self.vector_store.get_collection_info()
        except Exception as e:
            collection_info = {"error": str(e)}

        model_info = self.llm_client.get_model_info()

        status = {
            "llm_provider": model_info.get("provider"),
            "llm_model": model_info.get("model"),
            "vector_store": "chromadb",
            "collection_info": collection_info,
            "pdf_extractor": self.pdf_extractor.method,
            "chunk_size": self.text_chunker.chunk_size,
            "chunk_overlap": self.text_chunker.chunk_overlap,
        }

        if self.hybrid_retriever:
            status["hybrid_retriever"] = {
                "enabled": True,
                "retrieval_mode": self.retrieval_mode,
                "fusion_method": getattr(self.hybrid_retriever, "fusion_method", "weighted"),
                "dense_weight": getattr(self.hybrid_retriever, "dense_weight", 0.5),
                "bm25_weight": getattr(self.hybrid_retriever, "bm25_weight", 0.5),
            }

        if self.reranker and getattr(self.reranker, "model", None) is not None:
            candidate_k = (
                getattr(self.config.rerank, "candidate_top_k", 20)
                if (self.config and getattr(self.config, "rerank", None))
                else 20
            )
            final_k = (
                getattr(self.config.rerank, "final_top_k", 5)
                if (self.config and getattr(self.config, "rerank", None))
                else 5
            )
            status["reranker"] = {
                "enabled": True,
                "model": self.reranker.model_name,
                "candidate_top_k": candidate_k,
                "final_top_k": final_k,
            }
        else:
            status["reranker"] = {"enabled": False}

        if self.verifier is not None:
            status["verification"] = {
                "enabled": True,
                "temperature": getattr(self.verifier, "temperature", 0.0),
                "max_tokens": getattr(self.verifier, "max_tokens", 1024),
            }
        else:
            status["verification"] = {"enabled": False}

        return status
