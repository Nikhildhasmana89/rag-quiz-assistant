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

## Planned Experiments (Future Steps)

| Step | Experiment Name | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Baseline RAG** | Baseline stabilization & evaluation | **COMPLETED** |
| **Step 2** | Structure-Aware Document Processing | Section hierarchy & page metadata | *Next Up* |
| **Step 3** | Hybrid Retrieval | BM25 sparse + dense vector fusion | *Planned* |
| **Step 4** | Neural Reranking | Cross-Encoder top-K reranking | *Planned* |
| **Step 5** | Evidence Verification | NLI / Fact-checking verification | *Planned* |
| **Step 6** | Advanced Citations | Chunk & page-level citation anchors | *Planned* |
| **Step 7** | Multi-Document Comparison | Cross-document synthesis matrix | *Planned* |
| **Step 8** | Adaptive Learning & Quiz | Confidence scoring & quiz mastery | *Planned* |
| **Step 9** | Evaluation Dashboard | Real-time Ragas / RAG Triad suite | *Planned* |
| **Step 10**| Final Optimization & Packaging | Performance profiling & paper write-up | *Planned* |
