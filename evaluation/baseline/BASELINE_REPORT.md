# Baseline RAG Evaluation Report

**Project:** `rag-quiz-assistant`  
**Step:** Step 1 — Baseline Stabilization & Evaluation  
**Date:** September 2026  
**Document Evaluated:** `Nikhil_Dhasmana_Resume_2.pdf` (1 Page, 6 vector chunks)  
**Evaluator:** Baseline Automated Test Harness  

---

## 1. Current Architecture

The existing baseline RAG implementation operates as follows:

```text
PDF Document (pdfplumber)
         ↓
Text Extraction (Plain page strings, whitespace trimmed)
         ↓
Text Chunking (LangChain RecursiveCharacterTextSplitter: 500 chars, 0 overlap)
         ↓
Embedding Generation (all-MiniLM-L6-v2: 384-dimensional dense vectors)
         ↓
Vector Storage (ChromaDB PersistentClient saved to ./data/chroma)
         ↓
User Query Input
         ↓
Query Vectorization (Sentence Transformers via ChromaDB embedding function)
         ↓
Semantic Search (Approximate Nearest Neighbors via Squared Euclidean L2 distance)
         ↓
Top-K Context Retrieval (Default K=5 chunks)
         ↓
Strict Grounded Prompt Construction ("Answer using only the provided information...")
         ↓
LLM Inference (Groq LPU / OpenAI / Lamini)
         ↓
Answer + Source Chunks
```

---

## 2. Current Strengths

1. **Modular Architecture & Decoupling:** Clean separation of concerns with an abstract `LLMClient` factory allowing zero-code switching between Groq, OpenAI, and Lamini.
2. **Deterministic, Local Embeddings:** `all-MiniLM-L6-v2` runs entirely on the local host with zero external API latency, zero API costs, and consistent 384-dimensional representations.
3. **High-Speed Vector Search:** ChromaDB queries against small-to-medium documents execute in sub-100 millisecond time frames (average **45.0 ms** across our test suite).
4. **Sentence-Aware Boundary Chunking:** Using LangChain's `RecursiveCharacterTextSplitter` prevents arbitrary word clipping by hierarchically splitting on paragraphs, line breaks, sentences, and spaces.
5. **Strict Anti-Hallucination Prompting:** The prompt template explicitly instructs the LLM: *"If the information is not sufficient to answer the question, say so clearly."*

---

## 3. Current Limitations (Verified in Source Code)

1. **Pure Semantic Retrieval (No Hybrid / BM25 Search):**
   - ChromaDB relies entirely on dense vector dot products/Euclidean distance.
   - Exact alphanumeric tokens (e.g., specific phone numbers, course codes like "AKTU", dates like "October 2025") can have lower cosine similarity if the query uses different phrasing, causing keyword retrieval misses that BM25 would easily catch.
2. **Absence of Neural Reranking:**
   - The top $K$ chunks are passed directly into the LLM context in raw vector distance order without a Cross-Encoder reranker to verify token-level query-document alignment.
3. **Loss of Document Structure & Page Metadata:**
   - During PDF ingestion in `app/rag_pipeline.py`, page texts are concatenated and chunked. The metadata stored in ChromaDB records only `source`, `source_file`, and `chunk_size`.
   - **Page numbers and section headers are not preserved** in the stored chunk metadata, preventing precise page-level citations.
4. **ChromaDB Metadata Compatibility Issue:**
   - In `app/embeddings/chroma_store.py` line 107, default empty metadata was passed as `[{}]`, which raises a `ValueError` in newer ChromaDB releases (`chromadb>=1.5`) requiring `metadatas=None` when no metadata attributes are present.
5. **No Evidence Verification Layer:**
   - The system assumes that retrieved chunks are factually adequate; there is no independent NLI (Natural Language Inference) or entailment step checking whether the final generated answer is truly entailed by the context before displaying it to the user.

---

## 4. Baseline Evaluation Results

A benchmark dataset of **12 questions across 10 academic categories** was executed against the baseline vector retrieval engine.

### Quantitative Summary Table

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Questions Evaluated** | **12** | Covering factual, definition, methodology, numerical, results, limitations, contribution, comparison, multi-context, unanswerable |
| **Evidence Retrieval Rate** | **100% (12/12)** | Chunks containing the required ground truth were retrieved in the Top-3 results |
| **Average Retrieval Latency** | **0.0450 s (45.0 ms)** | Local CPU inference on ChromaDB index |
| **Fastest Query Latency** | **0.0306 s (30.6 ms)** | `q011` (Limitations) |
| **Slowest Query Latency** | **0.1007 s (100.7 ms)** | `q001` (First query cache warm-up) |
| **Average Chunk Length** | **427.6 characters** | Configured chunk size = 500 characters |
| **Total Stored Chunks** | **6 chunks** | 1-page test PDF |

---

## 5. Performance by Question Category

| Category | Question ID | Retrieval Time (s) | Evidence Found? | Key Retrieved Chunk |
| :--- | :--- | :--- | :--- | :--- |
| **Factual** | `q001` | 0.1007 | Yes | Chunk #6 (Education & College) |
| **Factual** | `q002` | 0.0351 | Yes | Chunk #6 (Certifications & Dates) |
| **Definition** | `q003` | 0.0477 | Yes | Chunk #3 & #4 (Project Tech Stack) |
| **Methodology** | `q004` | 0.0688 | Yes | Chunk #4 (JWT & RBAC Access Control) |
| **Methodology** | `q005` | 0.0466 | Yes | Chunk #4 (Socket.IO & WebSockets) |
| **Numerical** | `q006` | 0.0315 | Yes | Chunk #1 & #6 (Year 2027, Phone) |
| **Results** | `q007` | 0.0347 | Yes | Chunk #5 (RESTful APIs & Mongo Schemas) |
| **Contribution** | `q008` | 0.0343 | Yes | Chunk #2 & #5 (Next.js, Tailwind UI) |
| **Comparison** | `q009` | 0.0384 | Yes | Chunk #2 & #3 (MongoDB vs MySQL) |
| **Multi-Context** | `q010` | 0.0364 | Yes | Chunks #1, #2, #3, #4 (Skills + Project) |
| **Limitations** | `q011` | 0.0306 | Yes | Chunk #2 (Postman testing noted, no CI/CD) |
| **Unanswerable** | `q012` | 0.0352 | Yes | Correctly retrieves no conflicting claims |

---

## 6. Conclusion for Step 1 Baseline

The baseline system demonstrates stable, low-latency semantic retrieval on clean text documents. However, the retrieval relies entirely on dense embedding distance and lacks structural document metadata (section headings, page numbers). 

This baseline benchmark provides the exact quantitative foundation required for comparing the subsequent steps of the research project:
1. **Baseline Vector Search** (Current: 45 ms latency, semantic-only, chunk size 500)
2. **Step 2: Structure-Aware Document Processing + Metadata**
3. **Step 3: Hybrid Retrieval (BM25 + Dense Vectors)**
4. **Step 4: Neural Reranking**
5. **Step 5: Evidence Verification**
