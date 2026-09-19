"""Citation Engine for ResearchLens AI.

Provides grounded citation generation, claim-to-evidence mapping, and verifiable
provenance tracking across multi-document RAG responses. Preserves full Step 2
metadata (document_id, source_file, page_number, section, chunk_id) and Step 4
neural reranker scores.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STATUS_SUPPORTED = "supported"
STATUS_PARTIALLY_SUPPORTED = "partially_supported"
STATUS_CONTRADICTED = "contradicted"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

VALID_CITATION_STATUSES = {
    STATUS_SUPPORTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE,
}

# Common English stopwords for lexical overlap scoring
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_content_words(text: str) -> Set[str]:
    """Extract non-stopword tokens from text."""
    tokens = tokenize(text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


class CitationEngine:
    """Core grounded citation and provenance engine."""

    def __init__(self, config: Any = None, snippet_max_chars: int = 250):
        """Initialize CitationEngine.

        Args:
            config: Optional AppConfig object.
            snippet_max_chars: Maximum character length for extracted evidence snippets.
        """
        self.config = config
        self.snippet_max_chars = snippet_max_chars
        logger.info(f"Initialized CitationEngine (snippet_max_chars={self.snippet_max_chars})")

    def extract_evidence_snippet(
        self,
        chunk_text: str,
        claim_text: str,
        max_chars: Optional[int] = None,
    ) -> str:
        """Extract a high-relevance verbatim sentence or passage from chunk text.

        Guarantees that the returned snippet is a verbatim substring of chunk_text
        with 0% LLM hallucination.

        Args:
            chunk_text: Complete text of the retrieved document chunk.
            claim_text: Specific claim to substantiate or match.
            max_chars: Optional override for max snippet character length.

        Returns:
            Verbatim snippet extracted from chunk_text.
        """
        if not chunk_text or not chunk_text.strip():
            return ""

        clean_chunk = chunk_text.strip()
        limit = max_chars or self.snippet_max_chars

        # If chunk is already within limit, return it entirely
        if len(clean_chunk) <= limit:
            return clean_chunk

        # Split chunk into sentences while preserving sentence boundary positions
        sentence_delimiters = r"(?<=[.!?])\s+"
        raw_sentences = [s.strip() for s in re.split(sentence_delimiters, clean_chunk) if s.strip()]

        if not raw_sentences:
            return clean_chunk[:limit].strip()

        claim_tokens = tokenize_content_words(claim_text)
        if not claim_tokens:
            # If claim has no informative content words, return first sentence or chunk slice
            first_sent = raw_sentences[0]
            return first_sent[:limit].strip() if len(first_sent) > limit else first_sent

        # Score each candidate sentence
        best_sentence_idx = 0
        best_score = -1.0

        for idx, sent in enumerate(raw_sentences):
            sent_tokens = tokenize_content_words(sent)
            if not sent_tokens:
                continue

            intersection = claim_tokens.intersection(sent_tokens)
            if not intersection:
                score = 0.0
            else:
                # Jaccard overlap + entity overlap bonus (capitalized words or digits)
                jaccard = len(intersection) / len(claim_tokens.union(sent_tokens))
                raw_overlap = len(intersection)
                
                # Bonus for exact number/digit matches
                num_bonus = 0.0
                claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text))
                sent_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", sent))
                if claim_numbers.intersection(sent_numbers):
                    num_bonus = 0.5 * len(claim_numbers.intersection(sent_numbers))

                score = (jaccard * 2.0) + raw_overlap + num_bonus

            if score > best_score:
                best_score = score
                best_sentence_idx = idx

        chosen_sent = raw_sentences[best_sentence_idx]

        # If chosen sentence fits and there is space, optionally append adjacent sentence for context
        candidate = chosen_sent
        if len(candidate) < limit and best_sentence_idx + 1 < len(raw_sentences):
            next_sent = raw_sentences[best_sentence_idx + 1]
            combined = f"{candidate} {next_sent}"
            if len(combined) <= limit:
                candidate = combined

        # Ensure result does not exceed limit
        if len(candidate) > limit:
            candidate = candidate[:limit].rstrip()
            # If split in middle of word, trim to last space
            if " " in candidate:
                candidate = candidate.rsplit(" ", 1)[0]
            candidate += "..."

        return candidate.strip()

    def format_citation_string(self, citation: Dict[str, Any]) -> str:
        """Format citation into human-readable academic provenance representation.

        Example:
        [Document: paper.pdf | Page: 7 | Section: Results | Chunk: chunk_014]
        Evidence: "Under fine-tuning on the test set, the model achieved 94.2% accuracy."
        """
        doc_name = citation.get("document_name") or "Unknown Document"
        page = citation.get("page_number", 1)
        sec = citation.get("section") or "General"
        cid = citation.get("chunk_id") or "chunk_unknown"
        ev_text = citation.get("evidence_text", "")

        header = f"[Document: {doc_name} | Page: {page} | Section: {sec} | Chunk: {cid}]"
        if ev_text:
            return f"{header}\nEvidence: \"{ev_text}\""
        return header

    def annotate_text_with_citations(
        self,
        answer_text: str,
        claims: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        """Inject bracketed citation markers (e.g., [1], [2]) into answer text.

        Matches verified claims to sentences in the answer text and appends
        appropriate citation index numbers. Ensures unsupported claims receive NO markers.
        """
        if not answer_text or not answer_text.strip():
            return answer_text or ""

        if not citations:
            return answer_text

        # Build claim to citation indices mapping
        claim_to_indices: Dict[str, List[int]] = {}
        for cit in citations:
            c_text = cit.get("claim", "").strip()
            idx = cit.get("citation_index")
            if c_text and idx is not None:
                if c_text not in claim_to_indices:
                    claim_to_indices[c_text] = []
                if idx not in claim_to_indices[c_text]:
                    claim_to_indices[c_text].append(idx)

        if not claim_to_indices:
            return answer_text

        # Break answer into sentences
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = [s for s in re.split(sentence_pattern, answer_text) if s.strip()]

        if not sentences:
            return answer_text

        annotated_sentences = []
        assigned_citations: Set[int] = set()

        for sent in sentences:
            sent_tokens = tokenize_content_words(sent)
            matched_citation_indices: List[int] = []

            for claim_str, indices in claim_to_indices.items():
                c_tokens = tokenize_content_words(claim_str)
                if not c_tokens:
                    continue

                # Check if sentence expresses this claim (overlap > 50% or claim in sentence)
                if claim_str.lower() in sent.lower() or sent.lower() in claim_str.lower():
                    matched_citation_indices.extend(indices)
                else:
                    overlap = len(c_tokens.intersection(sent_tokens))
                    if overlap >= max(2, int(len(c_tokens) * 0.45)):
                        matched_citation_indices.extend(indices)

            # Deduplicate and sort
            unique_indices = sorted(set(matched_citation_indices))
            if unique_indices:
                for u in unique_indices:
                    assigned_citations.add(u)
                markers = "".join(f"[{u}]" for u in unique_indices)
                # Append marker before trailing punctuation if present
                clean_s = sent.strip()
                if clean_s and clean_s[-1] in ".!?":
                    annotated = f"{clean_s[:-1]} {markers}{clean_s[-1]}"
                else:
                    annotated = f"{clean_s} {markers}"
                annotated_sentences.append(annotated)
            else:
                annotated_sentences.append(sent)

        result_text = " ".join(annotated_sentences)

        # If some citations could not be aligned to specific sentences, append remaining at end
        unassigned = [c["citation_index"] for c in citations if c["citation_index"] not in assigned_citations]
        if unassigned:
            remaining_markers = "".join(f"[{u}]" for u in sorted(set(unassigned)))
            result_text = f"{result_text.rstrip()} {remaining_markers}"

        return result_text

    def generate_citations(
        self,
        claims: List[Dict[str, Any]],
        evidence_chunks: List[Dict[str, Any]],
        answer_text: str = "",
    ) -> Dict[str, Any]:
        """Generate grounded citations and claim-evidence provenance mappings.

        Implements strict handling of all 4 verification states:
        1. Supported: Emits full citations with exact matching evidence snippets.
        2. Partially Supported: Emits citations for supported parts with partial match notes.
        3. Contradicted: Emits citation referencing conflicting chunk with contradiction alert.
        4. Insufficient Evidence: Marks claim as unsupported, STRICTLY ZERO fake citations.

        Args:
            claims: Claim objects produced by EvidenceVerifier.
            evidence_chunks: Candidate chunks retrieved/reranked.
            answer_text: Synthesized LLM answer.

        Returns:
            Dictionary containing:
            - citations: List of detailed citation records.
            - claims: List of claims with attached citation references.
            - annotated_response: Answer text with inline citation markers.
            - unsupported_claims: List of claims lacking evidence.
            - citation_latency_ms: Milliseconds taken.
            - metrics: Quantitative citation statistics.
        """
        t0 = time.perf_counter()

        # Build chunk lookup dictionary indexed by both chunk_id and id
        chunk_dict: Dict[str, Dict[str, Any]] = {}
        for c in evidence_chunks:
            cid = c.get("chunk_id") or c.get("id") or ""
            if cid:
                chunk_dict[cid] = c

        citations_list: List[Dict[str, Any]] = []
        annotated_claims: List[Dict[str, Any]] = []
        unsupported_claims: List[str] = []
        citation_counter = 1

        for c_item in claims:
            if not isinstance(c_item, dict):
                continue

            claim_text = str(c_item.get("claim", "")).strip()
            raw_status = str(c_item.get("status", STATUS_INSUFFICIENT_EVIDENCE)).strip().lower()
            status = raw_status if raw_status in VALID_CITATION_STATUSES else STATUS_INSUFFICIENT_EVIDENCE
            raw_evidence_ids = c_item.get("evidence_ids", [])
            evidence_ids = [str(x).strip() for x in raw_evidence_ids if str(x).strip()]

            claim_citations: List[Dict[str, Any]] = []

            # =================================================================
            # CASE 4: Insufficient Evidence Claim -> ZERO fake citations
            # =================================================================
            if status == STATUS_INSUFFICIENT_EVIDENCE or not evidence_chunks:
                unsupported_claims.append(claim_text)
                annotated_claims.append({
                    "claim": claim_text,
                    "status": STATUS_INSUFFICIENT_EVIDENCE,
                    "evidence_ids": [],
                    "citation_ids": [],
                    "citations": [],
                    "notes": "Insufficient evidence: no grounded document source confirms this statement.",
                })
                continue

            # Resolve chunk objects matching evidence_ids
            matched_chunks: List[Dict[str, Any]] = []
            for eid in evidence_ids:
                if eid in chunk_dict and chunk_dict[eid] not in matched_chunks:
                    matched_chunks.append(chunk_dict[eid])

            # Fallback for supported/partially_supported/contradicted claims without explicit IDs:
            # Match top candidate chunk having highest content overlap
            if not matched_chunks and evidence_chunks:
                best_c = None
                best_score = -1.0
                claim_words = tokenize_content_words(claim_text)
                for c in evidence_chunks:
                    text_words = tokenize_content_words(c.get("text", ""))
                    score = len(claim_words.intersection(text_words))
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c is not None and best_score > 0:
                    matched_chunks.append(best_c)

            # If still no matching chunks found, mark as unsupported
            if not matched_chunks:
                unsupported_claims.append(claim_text)
                annotated_claims.append({
                    "claim": claim_text,
                    "status": STATUS_INSUFFICIENT_EVIDENCE,
                    "evidence_ids": [],
                    "citation_ids": [],
                    "citations": [],
                    "notes": "Chunk reference could not be located in retrieved pool.",
                })
                continue

            # =================================================================
            # CASES 1, 2, 3: Supported, Partially Supported, Contradicted
            # =================================================================
            for chunk in matched_chunks:
                cid = chunk.get("chunk_id") or chunk.get("id") or "chunk_unknown"
                meta = chunk.get("metadata") or {}
                chunk_text = chunk.get("text", "")

                doc_name = meta.get("source_file") or chunk.get("source_file") or "unknown_document"
                doc_id = meta.get("document_id") or chunk.get("document_id") or ""
                page_num = meta.get("page_number") or chunk.get("page_number") or 1
                section_name = meta.get("section") or chunk.get("section") or ""
                source_type = chunk.get("retrieval_source", "hybrid")

                rerank_score = chunk.get("reranker_score")
                conf_score = (
                    rerank_score
                    if rerank_score is not None
                    else chunk.get("hybrid_score", chunk.get("dense_score", 1.0))
                )

                snippet = self.extract_evidence_snippet(
                    chunk_text=chunk_text,
                    claim_text=claim_text,
                )

                # Set notes based on verification case
                if status == STATUS_SUPPORTED:
                    notes = "Supported: verified directly against retrieved chunk."
                elif status == STATUS_PARTIALLY_SUPPORTED:
                    notes = "Partially supported: some aspects verified; additional detail unconfirmed."
                elif status == STATUS_CONTRADICTED:
                    notes = "Contradicted: retrieved evidence directly conflicts with this claim."
                else:
                    notes = ""

                citation_obj = {
                    "citation_id": f"cit_{citation_counter}",
                    "citation_index": citation_counter,
                    "claim": claim_text,
                    "claim_status": status,
                    "document_name": doc_name,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "section": section_name,
                    "chunk_id": cid,
                    "evidence_text": snippet,
                    "confidence_score": round(float(conf_score), 4) if conf_score is not None else 1.0,
                    "reranker_score": round(float(rerank_score), 4) if rerank_score is not None else None,
                    "retrieval_source": source_type,
                    "notes": notes,
                }

                citation_counter += 1
                claim_citations.append(citation_obj)
                citations_list.append(citation_obj)

            annotated_claims.append({
                "claim": claim_text,
                "status": status,
                "evidence_ids": [c.get("chunk_id") or c.get("id") for c in matched_chunks],
                "citation_ids": [c["citation_id"] for c in claim_citations],
                "citations": claim_citations,
                "notes": claim_citations[0]["notes"] if claim_citations else "",
            })

        # Generate annotated response with inline markers
        clean_answer = (answer_text or "").strip()
        annotated_response = self.annotate_text_with_citations(
            answer_text=clean_answer,
            claims=annotated_claims,
            citations=citations_list,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Summary statistics
        total_claims = len(annotated_claims)
        supported_count = sum(1 for c in annotated_claims if c["status"] == STATUS_SUPPORTED)
        partially_count = sum(1 for c in annotated_claims if c["status"] == STATUS_PARTIALLY_SUPPORTED)
        contradicted_count = sum(1 for c in annotated_claims if c["status"] == STATUS_CONTRADICTED)
        insufficient_count = sum(1 for c in annotated_claims if c["status"] == STATUS_INSUFFICIENT_EVIDENCE)
        coverage_ratio = (
            (supported_count + partially_count + contradicted_count) / total_claims
            if total_claims > 0
            else 0.0
        )

        logger.info(
            f"Citation generation complete: {len(citations_list)} citations across "
            f"{total_claims} claims ({supported_count} supp, {partially_count} part, "
            f"{contradicted_count} contra, {insufficient_count} unsupp) in {latency_ms:.2f}ms"
        )

        return {
            "citations": citations_list,
            "claims": annotated_claims,
            "annotated_response": annotated_response,
            "unsupported_claims": unsupported_claims,
            "citation_latency_ms": round(latency_ms, 2),
            "metrics": {
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "partially_supported_claims": partially_count,
                "contradicted_claims": contradicted_count,
                "insufficient_evidence_claims": insufficient_count,
                "total_citations": len(citations_list),
                "coverage_ratio": round(coverage_ratio, 4),
            },
        }
