# Experiment Log: Evidence-Based RAG Research

This document tracks the iterative development and experimental evaluation of the RAG system across all planned project phases.

---

## Experiment 1: Baseline RAG

- **Date:** September 2026
- **System Version:** Baseline (Step 1)
- **Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 Chunks, ~2,560 characters)

### Configuration
- **Retrieval Engine:** Semantic Vector Search (ChromaDB approximate nearest neighbors via Squared Euclidean distance)
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions, local inference)
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 0 characters
- **Splitter:** LangChain `RecursiveCharacterTextSplitter`
- **Reranking:** Not implemented (Top-K raw vector results directly fed to LLM)
- **Evidence Verification:** Not implemented (No independent NLI / entailment verification layer)
- **Citation Granularity:** Basic document-level provenance (`source_file` only; page numbers and section headers not preserved in chunk metadata)
- **LLM Provider Supported:** Groq (`llama3-8b-8192`), OpenAI (`gpt-3.5-turbo`), Lamini

### Quantitative Benchmark Results
- **Questions Tested:** 12 questions (Factual, Definition, Methodology, Numerical, Results, Limitations, Contribution, Comparison, Multi-Context, Unanswerable)
- **Evidence Retrieval Rate:** 100% (12 / 12)
- **Average Retrieval Latency:** 0.0450 seconds (45.0 ms)
- **Fastest Query:** 0.0306 seconds (30.6 ms)
- **Slowest Query:** 0.1007 seconds (100.7 ms)

### Qualitative Observations
1. **Semantic Understanding:** The dense embedding model accurately mapped conceptual queries (e.g. "educational qualification" $\rightarrow$ college degree chunk; "tech stack" $\rightarrow$ project description chunk) even when wording differed from source text.
2. **Multi-Context Challenge:** When questions span multiple disjoint sections (e.g. comparing skills list with project implementation), the system relies on retrieving multiple independent chunks. Without reranking or structural aggregation, synthesis relies heavily on LLM context stitching.
3. **Absence of Keyword Matching:** Exact term searches (dates, course acronyms) succeed when surrounding semantic context is strong, but pure dense embeddings risk lower precision on isolated alphanumeric codes.

### Technical Limitations Identified
1. Pure semantic vector search with no sparse/BM25 keyword component.
2. No Cross-Encoder or neural reranking stage to re-order top-K candidates.
3. Metadata is minimal (`source`, `source_file`, `chunk_size`); page numbers and section headers are missing.
4. No post-generation evidence verification checking whether generated statements are entailed by the retrieved context.

---


---

## Experiment 2: Structure-Aware Document Processing & Metadata

- **Date:** September 2026
- **System Version:** Step 2 (Structure-Aware Document Processing & Rich Metadata)
- **Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 Chunks, 53,622 bytes)

### Configuration
- **Retrieval Engine:** Semantic Vector Search with Full Provenance (ChromaDB approximate nearest neighbors)
- **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions, local inference)
- **Document ID Strategy:** Deterministic SHA-256 hash (`doc_{hash[:12]}`)
- **Chunk ID Strategy:** Deterministic globally unique identifier (`f"{document_id}_p{page}_c{chunk_index}"`)
- **Section Detection Strategy:** Deterministic regex heuristic identifying academic/technical section headings (*Abstract, Introduction, Background, Related Work, Methodology, System Architecture, Experimental Setup, Results, Discussion, Conclusion, Limitations, References, Technical Skills, Education, Project Experience*)
- **ChromaDB Compatibility:** Sanitized primitive dictionary types; `metadatas=None` fallback for empty collections (fixes ChromaDB 1.5.9 schema constraint)
- **Dual Retrieval API:**
  - `retrieve(query, n_results=5)` -> `List[str]` (100% backward compatible)
  - `retrieve_with_metadata(query, n_results=5)` -> `List[Dict[str, Any]]` (contains text, metadata, id, distance)

### Quantitative Benchmark Results
- **Schema Completeness Rate:** 100% (6/6 chunks contain document_id, source_file, page_number, section, chunk_id, chunk_size, document_type)
- **Chunk ID Collision Rate:** 0.0% (0 duplicate IDs across indexing operations)
- **Ingestion Latency:** 2.67 seconds
- **Average Retrieval Latency (with Metadata):** 0.159 seconds (159.8 ms)
- **Test Suite Pass Rate:** 100% (56 / 56 tests passed)

### Qualitative Observations
1. **Provenance Granularity:** Every retrieved chunk now carries explicit source file, page number, and section tags, making answers fully traceable to specific document locations.
2. **Deterministic Stability:** Ingestion of identical documents produces identical `document_id` and `chunk_id`s, preventing ghost duplicates and unneeded database bloat.
3. **Foundation Ready:** The unified chunk representation now provides the required fields (`document_id`, `chunk_id`, `text`, `section`, `page_number`) needed for BM25 indexing in Step 3 and Cross-Encoder reranking in Step 4.

---

---

## Experiment 3: Hybrid Retrieval (Dense Vector + BM25 Lexical Fusion)

- **Date:** September 2026
- **System Version:** Step 3 (Hybrid Retrieval with Weighted Score Fusion & RRF)
- **Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 Chunks, Document ID: `doc_dda75b4545a4`)

### Configuration
- **Dual Retrieval Engines:**
  - **Dense Channel:** ChromaDB approximate nearest neighbors with `all-MiniLM-L6-v2` (384 dimensions, L2 Euclidean distance mapped to similarity: $S = 1/(1 + d)$).
  - **Sparse Channel:** Lexical keyword search via `LuceneBM25Okapi` ($k_1=1.5, b=0.75$) with strictly non-negative IDF: $\ln(1 + (N - n + 0.5) / (n + 0.5))$.
- **Fusion Modes Supported:**
  1. **Weighted Normalized Score Fusion:** Min-Max normalized score aggregation with configurable weights $\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}}$ (default $\alpha = 0.5$).
  2. **Reciprocal Rank Fusion (RRF):** Rank-level aggregation $\sum_{m \in \{D, B\}} \frac{1}{k + r_m(c)}$ with smoothing constant $k=60$.
- **Deduplication:** Chunks appearing in both retrieval streams are deduplicated by `chunk_id` and assigned source tag `retrieval_source: "hybrid"`.
- **Index Persistence:** Local JSON storage at `./data/bm25/bm25_index.json` supporting instant application restarts and automatic sync from Chroma collection.
- **Backward Compatibility:** Single-channel vector-only mode retained (`mode="vector"`).

### Quantitative Benchmark Results (12 Questions Tested)
- **Mean Precision@5:**
  - Vector-Only: 0.7000
  - Hybrid Weighted: 0.6833
  - **Hybrid RRF: 0.7000**
- **Mean Recall@5:**
  - **Vector-Only: 1.0000**
  - **Hybrid Weighted: 1.0000**
  - Hybrid RRF: 0.9167
- **Mean Reciprocal Rank (MRR):**
  - Vector-Only: 0.8750
  - **Hybrid Weighted: 0.9167 (+4.7% improvement)**
  - Hybrid RRF: 0.8750
- **Mean nDCG@5 (Ranking Quality):**
  - Vector-Only: 0.8829
  - Hybrid Weighted: 0.8986 (+1.8%)
  - **Hybrid RRF: 0.9289 (+5.2% improvement)**
- **Average Retrieval Latency:**
  - Vector-Only: 74.99 ms
  - Hybrid Weighted: 168.37 ms
  - **Hybrid RRF: 69.18 ms (fastest)**
- **Test Suite Pass Rate:** 100% (68 / 68 tests passing)

### Qualitative Observations
1. **Ranking Separation:** Hybrid RRF demonstrated superior ability to rank unambiguous, multi-channel matches at rank 1, producing a **+5.2% higher nDCG@5** than pure dense vector retrieval.
2. **Lexical Grounding:** Queries with exact dates (e.g. "October 2025", "November 2025"), credentials ("CRUD Operations in MongoDB"), or phone numbers received high lexical boosts from BM25, mitigating semantic drift.
3. **Small-Corpus Stability:** The Lucene-style IDF adjustment prevented zero and negative scores on small document collections, maintaining robust keyword scoring.

---

## Experiment 4: Neural Cross-Encoder Reranking (Two-Stage Retrieval)

- **Date:** September 2026
- **System Version:** Step 4 (Hybrid Candidate Retrieval + Cross-Encoder Reranking)
- **Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 Chunks, Document ID: `doc_dda75b4545a4`)
- **Evaluation Dataset:** `evaluation/datasets/baseline_questions.json` (12 questions across 10 categories)

### Configuration
- **Two-Stage Architecture:**
  1. **Candidate Generation Stage (Hybrid Retrieval):** Dual-channel Dense (`all-MiniLM-L6-v2`) + Sparse (`LuceneBM25Okapi`) with Reciprocal Rank Fusion ($k=60$) retrieving top candidate pool ($K_{\text{pool}}=20$).
  2. **Neural Reranking Stage:** Pretrained Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 6 transformer layers, 384 hidden dimensions, ~80 MB) performing full all-to-all cross-attention across query-passage pairs $(q, d_i)$.
- **Candidate Pool Size ($K_{\text{pool}}$):** 20
- **Final Top-K ($K_{\text{final}}$):** 5
- **Batch Size:** 16 (local CPU inference)
- **Score Assignment:** `reranker_score` (raw relevance logit) and `rerank_rank` (1-based position post-rerank).
- **Metadata & Retrieval Score Preservation:** 100% preservation of all Step 2 metadata (`document_id`, `source_file`, `page_number`, `section`, `chunk_id`, `chunk_index`, `chunk_size`, `document_type`) and Step 3 retrieval scores (`dense_score`, `bm25_score`, `hybrid_score`, `retrieval_source`).
- **Backward Compatibility:** Toggleable via `RERANKING_ENABLED=false` or `--no-rerank`; single-channel vector-only mode (`mode="vector"`) remains 100% operational.

### Quantitative Benchmark Results (12 Questions Tested)
- **Mean Precision@5:**
  - Vector-Only (Baseline): 0.7000
  - Hybrid RRF (Step 3): 0.7000
  - **Hybrid + Cross-Encoder (Step 4): 0.7000**
- **Mean Recall@5:**
  - Vector-Only: 1.0000
  - Hybrid RRF: 0.9167
  - **Hybrid + Cross-Encoder: 1.0000 (100% Recall recovered, +9.1% over Step 3 RRF)**
- **Mean Reciprocal Rank (MRR):**
  - Vector-Only: 0.8750
  - Hybrid RRF: 0.8750
  - **Hybrid + Cross-Encoder: 0.8750**
- **Mean nDCG@5:**
  - Vector-Only: 0.8829
  - Hybrid RRF: 0.9289
  - **Hybrid + Cross-Encoder: 0.8818**
- **Average Retrieval & Rerank Latency:**
  - Vector-Only: 53.78 ms
  - Hybrid RRF: 42.49 ms
  - **Hybrid + Cross-Encoder Total: 448.40 ms** (Candidate Retrieval: 40.83 ms, Neural Reranking: 407.56 ms)
- **Test Suite Pass Rate:** 100% (78 / 78 tests passing)

### Qualitative Observations
1. **Recall Recovery via Two-Stage Funnel:** Step 3 RRF suffered a slight recall dip (0.9167) when ranking reciprocal decay dropped marginal answer chunks past rank 5. Step 4's candidate pool expansion ($K_{\text{pool}}=20$) allowed the Cross-Encoder to evaluate a broader candidate set and elevate relevant chunks back into the top-5 window, recovering full **1.0000 Recall**.
2. **Logit Calibration & Discriminative Power:** The Cross-Encoder produces unconstrained logit scores that clearly delineate high-relevance chunks (scores $+2.5$ to $+6.2$) from peripheral or irrelevant chunks (negative scores down to $-9.8$).
3. **CPU Efficiency:** At ~407 ms reranking latency on consumer CPU for 6-20 passages, the two-stage pipeline easily fits within interactive search bounds (<500 ms) while completely eliminating external API calls.

---

## Experiment 5: Evidence Verification & Hallucination Detection (Post-Generation Integrity)

- **Date:** September 2026
- **System Version:** Step 5 (Two-Stage Retrieval + LLM Generation + Evidence Verification)
- **Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 Chunks, Document ID: `doc_dda75b4545a4`)
- **Evaluation Dataset:** `evaluation/datasets/step5_verification_cases.json` (10 test cases across 7 categories)

### Configuration
- **Three-Stage Pipeline Architecture:**
  1. **Candidate Retrieval Stage:** Dual-channel Hybrid Retrieval ($K_{\text{pool}}=20$) with Reciprocal Rank Fusion ($k=60$) followed by Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`) yielding top $K_{\text{final}}=5$ chunks with full Step 2 provenance.
  2. **Answer Generation Stage:** Grounded answer synthesis via `LLMClient` (`groq/compound-mini`).
  3. **Evidence Verification Stage:** Modular `EvidenceVerifier` decomposing generated responses into atomic claims, assessing claim entailment against XML-encapsulated evidence chunks, and computing deterministic verification status (`supported`, `partially_supported`, `contradicted`, `insufficient_evidence`).
- **Prompt Security:** Rigid 'Documents as Data' XML isolation preventing prompt injection or instruction overriding from document text.
- **Verification Parameters:** `temperature=0.0` (deterministic), `max_tokens=1024`.
- **Backward Compatibility:** Toggleable via `EVIDENCE_VERIFICATION_ENABLED=false` or `--no-verify` / `verify=False`.

### Quantitative Benchmark Results (10 Evaluation Cases)
- **Classification Accuracy:** **90.0% (9 / 10 cases matched ground-truth label)**
- **Verification Status Breakdown:**
  - `supported`: 5 cases (100% verified)
  - `partially_supported`: 2 cases (100% caught hallucinated tool/platform claims)
  - `contradicted`: 1 case (detected direct factual conflict)
  - `insufficient_evidence`: 2 cases (accurately flagged out-of-corpus details)
- **Latency Profile (Mean across 10 queries):**
  - Candidate Retrieval (Hybrid + Rerank): 500.71 ms
  - Answer Generation (LLM): 2,252.62 ms
  - Evidence Verification (LLM): 8,310.19 ms
  - Step 4 Total Latency: 2,753.33 ms
  - Step 5 Total Pipeline Latency: 11,063.51 ms
- **Test Suite Pass Rate:** 100% (92 / 92 unit and integration tests passing)

### Qualitative Observations
1. **Partial Support Precision:** When answers mixed factual truths (e.g. MongoDB, Next.js) with fabricated additions (e.g. Redis caching, AWS certifications), the verifier isolated the ungrounded claims and properly assigned `partially_supported` instead of accepting the answer blindly.
2. **Contradiction Catching:** Conflicting claims regarding authentication (session cookies vs JWT/RBAC) were successfully tagged as `contradicted`, triggering visual warning indicators on the interface.
3. **Research-Honest Framing:** Output strictly avoids absolute claims of "100% hallucination-free", framing answers as verifiable statements grounded in retrieved evidence.

---

## Planned Experiments (Future Steps)

| Step | Experiment Name | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Baseline RAG** | Baseline stabilization & evaluation | **COMPLETED** |
| **Step 2** | **Structure-Aware Document Processing** | Section hierarchy & page metadata | **COMPLETED** |
| **Step 3** | **Hybrid Retrieval** | BM25 sparse + dense vector fusion | **COMPLETED** |
| **Step 4** | **Neural Reranking** | Cross-Encoder top-K reranking | **COMPLETED** |
| **Step 5** | **Evidence Verification** | Claim-level entailment & hallucination detection | **COMPLETED** |
| **Step 6** | Advanced Citations | Chunk & page-level citation anchors | *Next Up* |
| **Step 7** | Multi-Document Comparison | Cross-document synthesis matrix | *Planned* |
| **Step 8** | Adaptive Learning & Quiz | Confidence scoring & quiz mastery | *Planned* |
| **Step 9** | Evaluation Dashboard | Real-time Ragas / RAG Triad suite | *Planned* |
| **Step 10**| Final Optimization & Packaging | Performance profiling & paper write-up | *Planned* |


