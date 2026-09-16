"""Hybrid Retriever combining Dense Vector Search and BM25 Keyword Search.

Supports Weighted Normalized Score Fusion and Reciprocal Rank Fusion (RRF),
with deduplication and metadata preservation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize a list of scores to the range [0.0, 1.0].

    If all scores are equal, returns 1.0 if score > 0 else 0.0.

    Args:
        scores: List of float scores

    Returns:
        List of normalized float scores
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0 if max_s > 0 else 0.0 for _ in scores]

    range_s = max_s - min_s
    return [(s - min_s) / range_s for s in scores]


class HybridRetriever:
    """Combines Dense Vector Retrieval (ChromaDB) with Lexical BM25 Retrieval."""

    def __init__(
        self,
        vector_store,
        bm25_retriever,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        fusion_method: str = "weighted",
        rrf_k: int = 60,
    ):
        """Initialize the HybridRetriever.

        Args:
            vector_store: ChromaStore instance
            bm25_retriever: BM25Retriever instance
            dense_top_k: Number of candidates to retrieve from dense store
            bm25_top_k: Number of candidates to retrieve from BM25
            final_top_k: Number of final fused candidates to return
            dense_weight: Weight for dense scores in weighted fusion
            bm25_weight: Weight for BM25 scores in weighted fusion
            fusion_method: 'weighted' or 'rrf'
            rrf_k: Smoothing constant for Reciprocal Rank Fusion
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)
        self.fusion_method = fusion_method.lower()
        self.rrf_k = int(rrf_k)

    def retrieve_dense(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates from dense vector store.

        Args:
            query: Query string
            n_results: Maximum candidates to retrieve (defaults to self.dense_top_k)

        Returns:
            List of candidate dictionaries:
            [{'chunk_id': ..., 'text': ..., 'metadata': ..., 'dense_score': ..., 'dense_rank': ...}]
        """
        k = n_results if n_results is not None else self.dense_top_k

        if self.vector_store is None:
            return []

        if getattr(self.vector_store, "collection", None) is None:
            try:
                self.vector_store.create_collection()
            except Exception as e:
                logger.warning(f"Could not create collection in vector store: {e}")
                return []

        try:
            results = self.vector_store.query(
                query_texts=[query],
                n_results=k,
            )
        except Exception as e:
            logger.error(f"Dense vector query failed: {e}")
            return []

        if not results or "documents" not in results or not results["documents"]:
            return []

        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        candidates = []
        for rank, (doc, meta, cid, dist) in enumerate(zip(docs, metas, ids, dists), start=1):
            # Chroma returns L2 distance by default (0 = identical, larger = more distant)
            # Convert to similarity score in (0, 1]
            dist_val = float(dist) if dist is not None else 0.0
            similarity = 1.0 / (1.0 + max(0.0, dist_val))

            chunk_id = cid or (meta.get("chunk_id") if isinstance(meta, dict) else "")
            candidates.append({
                "chunk_id": chunk_id,
                "text": doc,
                "metadata": meta or {},
                "dense_score": similarity,
                "distance": dist_val,
                "dense_rank": rank,
            })

        return candidates

    def retrieve_bm25(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates from BM25 index.

        Args:
            query: Query string
            n_results: Maximum candidates to retrieve (defaults to self.bm25_top_k)

        Returns:
            List of candidate dictionaries
        """
        k = n_results if n_results is not None else self.bm25_top_k
        if not self.bm25_retriever:
            return []

        try:
            return self.bm25_retriever.query(query, n_results=k)
        except Exception as e:
            logger.error(f"BM25 query failed: {e}")
            return []

    def fuse_weighted(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform Weighted Normalized Score Fusion on dense and BM25 results.

        Formula:
            normalized_dense = min_max_normalize(dense_scores)
            normalized_bm25 = min_max_normalize(bm25_scores)
            fused_score = alpha * norm_dense + (1 - alpha) * norm_bm25

        Args:
            dense_results: List of dense candidate dicts
            bm25_results: List of BM25 candidate dicts
            alpha: Dense weight in [0, 1]. Defaults to self.dense_weight / (dense_weight + bm25_weight)

        Returns:
            Deduplicated list of fused candidate dicts sorted by hybrid_score descending
        """
        total_w = self.dense_weight + self.bm25_weight
        eff_alpha = (self.dense_weight / total_w) if total_w > 0 else 0.5
        if alpha is not None:
            eff_alpha = float(alpha)
        beta = 1.0 - eff_alpha

        # Normalize dense scores
        raw_dense_scores = [c["dense_score"] for c in dense_results]
        norm_dense_scores = min_max_normalize(raw_dense_scores)
        for cand, norm_s in zip(dense_results, norm_dense_scores):
            cand["norm_dense_score"] = norm_s

        # Normalize BM25 scores
        raw_bm25_scores = [c["bm25_score"] for c in bm25_results]
        norm_bm25_scores = min_max_normalize(raw_bm25_scores)
        for cand, norm_s in zip(bm25_results, norm_bm25_scores):
            cand["norm_bm25_score"] = norm_s

        # Map by chunk_id
        merged: Dict[str, Dict[str, Any]] = {}

        # Process dense candidates
        for c in dense_results:
            cid = c["chunk_id"]
            merged[cid] = {
                "chunk_id": cid,
                "text": c["text"],
                "metadata": c.get("metadata", {}),
                "dense_score": c["dense_score"],
                "norm_dense_score": c["norm_dense_score"],
                "dense_rank": c.get("dense_rank"),
                "bm25_score": 0.0,
                "norm_bm25_score": 0.0,
                "bm25_rank": None,
                "retrieval_source": "dense",
            }

        # Process BM25 candidates
        for c in bm25_results:
            cid = c["chunk_id"]
            if cid in merged:
                # Appeared in both: mark hybrid and populate BM25 fields
                merged[cid]["bm25_score"] = c["bm25_score"]
                merged[cid]["norm_bm25_score"] = c["norm_bm25_score"]
                merged[cid]["bm25_rank"] = c.get("bm25_rank")
                merged[cid]["retrieval_source"] = "hybrid"
                # Combine metadata if needed
                if not merged[cid]["metadata"] and c.get("metadata"):
                    merged[cid]["metadata"] = c["metadata"]
            else:
                # Appeared only in BM25
                merged[cid] = {
                    "chunk_id": cid,
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "dense_score": 0.0,
                    "norm_dense_score": 0.0,
                    "dense_rank": None,
                    "bm25_score": c["bm25_score"],
                    "norm_bm25_score": c["norm_bm25_score"],
                    "bm25_rank": c.get("bm25_rank"),
                    "retrieval_source": "bm25",
                }

        # Calculate hybrid score
        for c in merged.values():
            c["hybrid_score"] = (
                eff_alpha * c["norm_dense_score"] + beta * c["norm_bm25_score"]
            )

        # Sort descending by hybrid_score
        fused = list(merged.values())
        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)

        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank

        return fused

    def fuse_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Perform Reciprocal Rank Fusion (RRF) on dense and BM25 results.

        Formula:
            RRF_score(chunk) = sum_{m in {dense, bm25}} 1 / (k + rank_m(chunk))

        Args:
            dense_results: List of dense candidate dicts
            bm25_results: List of BM25 candidate dicts
            k: Smoothing constant (defaults to self.rrf_k = 60)

        Returns:
            Deduplicated list of candidate dicts sorted by hybrid_score descending
        """
        rrf_constant = k if k is not None else self.rrf_k
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, c in enumerate(dense_results, start=1):
            cid = c["chunk_id"]
            dense_rrf = 1.0 / (rrf_constant + rank)
            merged[cid] = {
                "chunk_id": cid,
                "text": c["text"],
                "metadata": c.get("metadata", {}),
                "dense_score": c.get("dense_score", 0.0),
                "dense_rank": rank,
                "bm25_score": 0.0,
                "bm25_rank": None,
                "rrf_dense": dense_rrf,
                "rrf_bm25": 0.0,
                "retrieval_source": "dense",
            }

        for rank, c in enumerate(bm25_results, start=1):
            cid = c["chunk_id"]
            bm25_rrf = 1.0 / (rrf_constant + rank)
            if cid in merged:
                merged[cid]["bm25_score"] = c.get("bm25_score", 0.0)
                merged[cid]["bm25_rank"] = rank
                merged[cid]["rrf_bm25"] = bm25_rrf
                merged[cid]["retrieval_source"] = "hybrid"
                if not merged[cid]["metadata"] and c.get("metadata"):
                    merged[cid]["metadata"] = c["metadata"]
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "dense_score": 0.0,
                    "dense_rank": None,
                    "bm25_score": c.get("bm25_score", 0.0),
                    "bm25_rank": rank,
                    "rrf_dense": 0.0,
                    "rrf_bm25": bm25_rrf,
                    "retrieval_source": "bm25",
                }

        # Calculate final RRF score
        for c in merged.values():
            c["hybrid_score"] = c["rrf_dense"] + c["rrf_bm25"]

        # Sort descending by hybrid_score
        fused = list(merged.values())
        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)

        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank

        return fused

    def retrieve_hybrid(
        self,
        query: str,
        n_results: Optional[int] = None,
        fusion_method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute complete hybrid retrieval pipeline: Dense + BM25 + Fusion.

        Args:
            query: Query string
            n_results: Number of final results to return (defaults to self.final_top_k)
            fusion_method: 'weighted' or 'rrf' (defaults to self.fusion_method)

        Returns:
            List of top-ranked fused candidate dicts
        """
        top_k = n_results if n_results is not None else self.final_top_k
        method = (fusion_method or self.fusion_method).lower()

        # Retrieve candidates
        dense_candidates = self.retrieve_dense(query, n_results=self.dense_top_k)
        bm25_candidates = self.retrieve_bm25(query, n_results=self.bm25_top_k)

        # Handle edge cases where one or both pools are empty
        if not dense_candidates and not bm25_candidates:
            return []

        if not dense_candidates:
            # Only BM25 candidates available
            for rank, c in enumerate(bm25_candidates, start=1):
                c["rank"] = rank
                c["hybrid_score"] = c.get("bm25_score", 0.0)
                c["retrieval_source"] = "bm25"
            return bm25_candidates[:top_k]

        if not bm25_candidates:
            # Only dense candidates available
            for rank, c in enumerate(dense_candidates, start=1):
                c["rank"] = rank
                c["hybrid_score"] = c.get("dense_score", 0.0)
                c["retrieval_source"] = "dense"
            return dense_candidates[:top_k]

        # Apply selected fusion method
        if method == "rrf":
            fused = self.fuse_rrf(dense_candidates, bm25_candidates, k=self.rrf_k)
        else:
            fused = self.fuse_weighted(dense_candidates, bm25_candidates)

        return fused[:top_k]

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        mode: str = "hybrid",
        fusion_method: Optional[str] = None,
    ) -> List[str]:
        """Convenience method returning list of chunk texts.

        Args:
            query: Search query
            n_results: Number of chunks to return
            mode: 'hybrid' or 'vector'
            fusion_method: 'weighted' or 'rrf' (only used in hybrid mode)

        Returns:
            List of chunk text strings
        """
        results = self.retrieve_with_metadata(
            query=query,
            n_results=n_results,
            mode=mode,
            fusion_method=fusion_method,
        )
        return [r["text"] for r in results if "text" in r]

    def retrieve_with_metadata(
        self,
        query: str,
        n_results: Optional[int] = None,
        mode: str = "hybrid",
        fusion_method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates with provenance metadata.

        Args:
            query: Search query
            n_results: Number of chunks to return
            mode: 'hybrid' or 'vector'
            fusion_method: 'weighted' or 'rrf'

        Returns:
            List of candidate dictionaries
        """
        top_k = n_results if n_results is not None else self.final_top_k

        if mode.lower() == "vector":
            dense_candidates = self.retrieve_dense(query, n_results=top_k)
            for rank, c in enumerate(dense_candidates, start=1):
                c["rank"] = rank
                c["hybrid_score"] = c.get("dense_score", 0.0)
                c["retrieval_source"] = "vector"
            return dense_candidates[:top_k]

        return self.retrieve_hybrid(
            query=query,
            n_results=top_k,
            fusion_method=fusion_method,
        )
