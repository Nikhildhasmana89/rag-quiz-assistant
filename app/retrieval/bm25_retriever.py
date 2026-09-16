"""BM25 Lexical Keyword Retriever for RAG pipeline.

This module provides keyword-based lexical search using Lucene-style BM25Okapi,
local JSON persistence, and synchronization with the Chroma vector store.
"""

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

try:
    from rank_bm25 import BM25Okapi

    class LuceneBM25(BM25Okapi):
        """BM25Okapi with Lucene's non-negative IDF formula: ln(1 + (N - n + 0.5) / (n + 0.5))."""

        def _calc_idf(self, nd):
            self.idf = {}
            for word, freq in nd.items():
                self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

except ImportError:
    BM25Okapi = None  # type: ignore
    LuceneBM25 = None  # type: ignore

logger = logging.getLogger(__name__)


def default_tokenizer(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words.

    Args:
        text: Input string to tokenize

    Returns:
        List of lowercase tokens
    """
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class BM25Retriever:
    """Lexical keyword retriever powered by LuceneBM25."""

    def __init__(
        self,
        persist_path: Optional[Union[str, Path]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer=None,
    ):
        """Initialize the BM25Retriever.

        Args:
            persist_path: Optional path to JSON file for index persistence
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            tokenizer: Optional custom tokenization function
        """
        if BM25Okapi is None:
            raise ImportError(
                "rank_bm25 package is required. Install via `pip install rank-bm25`"
            )

        self.persist_path = Path(persist_path) if persist_path else None
        self.k1 = float(k1)
        self.b = float(b)
        self.tokenizer = tokenizer or default_tokenizer

        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[LuceneBM25] = None
        self._id_to_index: Dict[str, int] = {}

    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """Build or rebuild BM25 index from a list of chunk dicts.

        Args:
            chunks: List of dictionaries with 'chunk_id', 'text', and optional 'metadata'

        Returns:
            Number of indexed chunks
        """
        self.clear_index()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            tokens = self.tokenizer(text)
            self._id_to_index[chunk_id] = len(self.corpus_chunks)
            self.corpus_chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
            })
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = LuceneBM25(self.tokenized_corpus, k1=self.k1, b=self.b)
            logger.info(f"Built BM25 index with {len(self.corpus_chunks)} chunks")
        else:
            self.bm25 = None
            logger.info("Empty corpus provided; BM25 index is cleared")

        return len(self.corpus_chunks)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Add or update documents in the BM25 index.

        Args:
            chunks: List of chunk dictionaries to add/update

        Returns:
            Total number of chunks in the updated corpus
        """
        chunk_map = {c["chunk_id"]: c for c in self.corpus_chunks}

        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            chunk_map[cid] = {
                "chunk_id": cid,
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
            }

        all_chunks = list(chunk_map.values())
        return self.build_index(all_chunks)

    def clear_index(self) -> None:
        """Clear the in-memory index and corpus."""
        self.corpus_chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._id_to_index = {}
        logger.debug("Cleared BM25 index")

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query the BM25 index.

        Args:
            query_text: Search query
            n_results: Maximum number of results to return
            min_score: Minimum BM25 score threshold (defaults to > 0.0)

        Returns:
            List of result dicts sorted by score descending:
            [{'chunk_id': ..., 'text': ..., 'metadata': ..., 'bm25_score': float, 'bm25_rank': int}]
        """
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokens = self.tokenizer(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Pair scores with chunk objects
        scored_results = []
        for idx, score in enumerate(scores):
            score_val = float(score)
            if score_val > min_score:
                chunk = self.corpus_chunks[idx]
                scored_results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "bm25_score": score_val,
                })

        # Sort descending by bm25_score
        scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)

        # Slice to n_results and assign rank
        top_results = scored_results[:n_results]
        for rank, item in enumerate(top_results, start=1):
            item["bm25_rank"] = rank

        return top_results

    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Persist corpus and index metadata to a JSON file.

        Args:
            path: Target JSON file path (defaults to self.persist_path)

        Returns:
            True if saved successfully
        """
        save_path = Path(path) if path else self.persist_path
        if not save_path:
            logger.warning("No persist path specified for BM25 save")
            return False

        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(self.corpus_chunks),
            "chunks": self.corpus_chunks,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved BM25 index with {len(self.corpus_chunks)} chunks to {save_path}")
        return True

    def load(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load corpus and rebuild BM25 index from a JSON file.

        Args:
            path: Target JSON file path (defaults to self.persist_path)

        Returns:
            True if successfully loaded, False otherwise
        """
        load_path = Path(path) if path else self.persist_path
        if not load_path or not load_path.exists():
            logger.debug(f"BM25 index file not found at {load_path}")
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = data.get("chunks", [])
            self.k1 = float(data.get("k1", self.k1))
            self.b = float(data.get("b", self.b))
            self.build_index(chunks)
            logger.info(f"Loaded BM25 index from {load_path} with {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {load_path}: {e}")
            return False

    def sync_from_vector_store(self, vector_store) -> int:
        """Synchronize BM25 index with chunks currently stored in Chroma vector store.

        Args:
            vector_store: ChromaStore instance

        Returns:
            Number of chunks synchronized
        """
        if vector_store is None:
            logger.warning("Vector store is None; skipping sync")
            return 0

        try:
            if vector_store.collection is None:
                vector_store.create_collection()

            all_data = vector_store.collection.get()
            ids = all_data.get("ids") or []
            documents = all_data.get("documents") or []
            metadatas = all_data.get("metadatas") or []

            chunks = []
            for cid, doc, meta in zip(ids, documents, metadatas):
                chunks.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta or {},
                })

            count = self.build_index(chunks)
            if self.persist_path:
                self.save()
            logger.info(f"Synced {count} chunks from Chroma store to BM25 index")
            return count
        except Exception as e:
            logger.error(f"Error syncing BM25 index from vector store: {e}")
            return 0
