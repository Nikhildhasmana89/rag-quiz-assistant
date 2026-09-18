"""Neural Cross-Encoder Reranker for retrieved candidates.

Uses a pretrained Cross-Encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)
to score joint (query, passage) relevance, re-ordering candidate pools
while strictly preserving Step 2 metadata and Step 3 retrieval scores.
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker powered by sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize Cross-Encoder model.

        Args:
            model_name: Pretrained model name from Hugging Face
            batch_size: Batch size for cross-encoder inference
            device: 'cpu', 'cuda', or None for auto-detection
            max_length: Maximum sequence length for token pairs
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.model = None

        try:
            from sentence_transformers import CrossEncoder

            t0 = time.perf_counter()
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            load_time = time.perf_counter() - t0
            logger.info(
                f"Loaded CrossEncoder model '{self.model_name}' in {load_time:.2f}s"
            )
        except Exception as e:
            logger.warning(
                f"Could not load CrossEncoder model '{self.model_name}': {e}. "
                "Reranker will operate in fallback pass-through mode."
            )
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks based on Cross-Encoder joint relevance scoring.

        Args:
            query: User search query
            candidates: List of candidate chunk dictionaries
            top_k: Number of top reranked results to return

        Returns:
            List of top_k reranked candidate dictionaries with reranker_score and rerank_rank
        """
        if not candidates:
            return []

        # Graceful fallback if query is empty or model failed to initialize
        if not query or not str(query).strip() or self.model is None:
            logger.debug("Bypassing neural reranking (empty query or model unavailable)")
            fallback_results = []
            for rank, cand in enumerate(candidates[:top_k], start=1):
                item = copy.deepcopy(cand)
                item["initial_rank"] = cand.get("rank", rank)
                item["reranker_score"] = float(cand.get("hybrid_score", cand.get("dense_score", 0.0)))
                item["rerank_rank"] = rank
                item["rank"] = rank
                if "distance" not in item:
                    item["distance"] = cand.get("distance", 0.0)
                fallback_results.append(item)
            return fallback_results

        try:
            # Prepare (query, document) pairs
            pairs = [[str(query), str(c.get("text", ""))] for c in candidates]

            t0 = time.perf_counter()
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            rerank_time_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                f"Reranked {len(candidates)} candidates in {rerank_time_ms:.2f}ms"
            )

            # Deep copy and enrich each candidate
            reranked = []
            for rank_idx, (cand, score) in enumerate(zip(candidates, scores), start=1):
                item = copy.deepcopy(cand)
                score_val = float(score)

                # Preserve and enrich all metadata & scores
                item["initial_rank"] = cand.get("rank", rank_idx)
                item["reranker_score"] = score_val
                if "distance" not in item:
                    item["distance"] = cand.get("distance", 0.0)
                reranked.append(item)

            # Sort descending by cross-encoder relevance score
            reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

            # Assign rerank_rank and active rank
            for final_rank, item in enumerate(reranked, start=1):
                item["rerank_rank"] = final_rank
                item["rank"] = final_rank

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {e}. Falling back to candidate pool ordering.")
            fallback_results = []
            for rank, cand in enumerate(candidates[:top_k], start=1):
                item = copy.deepcopy(cand)
                item["initial_rank"] = cand.get("rank", rank)
                item["reranker_score"] = float(cand.get("hybrid_score", cand.get("dense_score", 0.0)))
                item["rerank_rank"] = rank
                item["rank"] = rank
                if "distance" not in item:
                    item["distance"] = cand.get("distance", 0.0)
                fallback_results.append(item)
            return fallback_results
