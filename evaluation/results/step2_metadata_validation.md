# Step 2 Validation Report: Structure-Aware Document Processing & Metadata

- **Date:** September 2026
- **System Version:** Step 2 (Structure-Aware Ingestion & Rich Provenance Metadata)
- **Document Tested:** `data/input/Nikhil_Dhasmana_Resume_2.pdf` (53,622 bytes, 1 page)
- **Status:** **VERIFIED & COMPLETED** (100% Schema Completeness, 0 Chunk ID Collisions)

---

## 1. Objectives & Executive Summary

In Step 1 (Baseline), document ingestion flattened all extracted pages into a single unindexed stream of chunks. Chunks received sequential integer IDs (`"0"`, `"1"`, `"2"`), document-level provenance was lost, page numbers were discarded, and re-indexing caused chunk collisions or overwrite.

In **Step 2**, we implemented structure-aware document processing:
1. **Deterministic Document ID**: Computed from document binary content (`doc_{sha256[:12]}`).
2. **Page Retention**: Chunks are processed per physical page, preserving `page_number` (1-indexed).
3. **Deterministic, Globally Unique Chunk IDs**: `f"{document_id}_p{page_number}_c{chunk_index}"` preventing ID collisions across multi-document indexing.
4. **Deterministic Section Heading Detection**: Regex-based heuristic recognizing common academic and technical headings (*Abstract, Introduction, Background, Related Work, Methodology, System Architecture, Experimental Setup, Results, Discussion, Conclusion, Limitations, References, Technical Skills, Education, Project Experience*).
5. **Provenance Retrieval**: Extended `RAGPipeline` with `retrieve_with_metadata()` returning chunk texts paired with complete provenance dictionaries (`document_id`, `source_file`, `page_number`, `section`, `chunk_id`, `distance`), while preserving 100% backward compatibility for existing callers (`retrieve()` returning `List[str]`).
6. **ChromaDB 1.5.9 Compatibility**: Resolved strict schema constraints where empty metadata dicts `[{}]` and `None` values caused runtime exceptions.

---

## 2. Ingested Metadata Schema

| Field Name | Type | Invariant / Validation Rule | Example Value |
| :--- | :--- | :--- | :--- |
| `document_id` | `str` | Format: `doc_[0-9a-f]{12}`, non-empty, deterministic | `"doc_dda75b4545a4"` |
| `source_file` | `str` | File name matching input PDF | `"Nikhil_Dhasmana_Resume_2.pdf"` |
| `page_number` | `int` | Physical page index $\ge 1$ | `1` |
| `section` | `str` | Recognized section heading or `""` (never `None`) | `"Technical Skills"` |
| `chunk_id` | `str` | Deterministic format: `{doc_id}_p{page}_c{index}` | `"doc_dda75b4545a4_p1_c2"` |
| `chunk_index` | `int` | Sequential position within document $\ge 0$ | `2` |
| `chunk_size` | `int` | Character length of text chunk ($> 0$) | `470` |
| `document_type` | `str` | Normalized document format identifier | `"pdf"` |

---

## 3. Empirical Validation Results

### Ingestion Metrics
- **Source File:** `Nikhil_Dhasmana_Resume_2.pdf`
- **Pages Extracted:** 1
- **Chunks Generated:** 6
- **Document ID Assigned:** `doc_dda75b4545a4`
- **Ingestion Latency:** 2.67 seconds (including ONNX/SentenceTransformer embedding)

### Vector Store Chunk Audit
| Chunk Index | Chunk ID | Page | Section Detected | Chunk Size | Document ID |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Chunk #1 | `doc_dda75b4545a4_p1_c0` | 1 | `""` (Header/Summary) | 494 chars | `doc_dda75b4545a4` |
| Chunk #2 | `doc_dda75b4545a4_p1_c1` | 1 | `""` (Summary / Pre-skills) | 476 chars | `doc_dda75b4545a4` |
| Chunk #3 | `doc_dda75b4545a4_p1_c2` | 1 | **Technical Skills** | 470 chars | `doc_dda75b4545a4` |
| Chunk #4 | `doc_dda75b4545a4_p1_c3` | 1 | **Project Experience** | 435 chars | `doc_dda75b4545a4` |
| Chunk #5 | `doc_dda75b4545a4_p1_c4` | 1 | **Project Experience** | 454 chars | `doc_dda75b4545a4` |
| Chunk #6 | `doc_dda75b4545a4_p1_c5` | 1 | **Education** | 286 chars | `doc_dda75b4545a4` |

### Integrity & Completeness Statistics
| Validation Metric | Target | Measured | Result |
| :--- | :---: | :---: | :---: |
| **Total Chunks Stored** | 6 | 6 | **PASS** |
| **Unique Chunk IDs** | 6 / 6 | 6 / 6 | **PASS (0 Collisions)** |
| **Document ID Completeness** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Source File Completeness** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Page Number Completeness** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Section Field Presence** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Chunk ID Format Validity** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Chunk Size Exactness** | 100% | 6 / 6 (100.0%) | **PASS** |
| **Document Type Accuracy** | 100% | 6 / 6 (100.0%) | **PASS** |

---

## 4. Provenance Retrieval Verification

Queries were executed using `RAGPipeline.retrieve_with_metadata()`:

### Query 1: *"What are Nikhil's technical skills and programming languages?"*
- **Retrieval Latency:** 174.8 ms
- **Rank 1 Evidence:**
  - `Source:` `Nikhil_Dhasmana_Resume_2.pdf`
  - `Page Number:` 1
  - `Section:` `Education`
  - `Distance / Score:` 0.6848
  - `Snippet:` *"Inderprastha Engineering College (IPEC), Ghaziabad - B.Tech, Artificial Intelligence & Machine Learning (AIML)..."*
- **Rank 2 Evidence:**
  - `Source:` `Nikhil_Dhasmana_Resume_2.pdf`
  - `Page Number:` 1
  - `Section:` `(None detected)`
  - `Distance / Score:` 0.7103
  - `Snippet:` *"NIKHIL DHASMANA ... Full-Stack Developer specializing in the MERN stack, Next.js, and TypeScript..."*

### Query 2: *"What educational institution did Nikhil attend?"*
- **Retrieval Latency:** 144.9 ms
- **Rank 1 Evidence:**
  - `Source:` `Nikhil_Dhasmana_Resume_2.pdf`
  - `Page Number:` 1
  - `Section:` `(None detected)`
  - `Distance / Score:` 0.7565
- **Rank 2 Evidence:**
  - `Source:` `Nikhil_Dhasmana_Resume_2.pdf`
  - `Page Number:` 1
  - `Section:` `Education`
  - `Distance / Score:` 0.7729
  - `Snippet:` *"Inderprastha Engineering College (IPEC), Ghaziabad - B.Tech, Artificial Intelligence & Machine Learning (AIML)..."*

---

## 5. Backward Compatibility & System Integrity

1. **`pipeline.retrieve(query, n_results=5)`**:
   - Signature: Returns `List[str]`.
   - Verified: Exactly matches the Step 1 interface; frontend and existing scripts execute without modification.
2. **`pipeline.retrieve_with_metadata(query, n_results=5)`**:
   - Signature: Returns `List[Dict[str, Any]]` containing `{"text", "metadata", "id", "distance"}`.
   - Verified: Enables downstream features (Hybrid Search, Reranking, Evidence Verification).
3. **`pipeline.rag_query(query, ...)`**:
   - Returned Dict contains:
     - `"query"`: original question
     - `"retrieved_documents"`: `List[str]` (backward-compatible)
     - `"retrieved_chunks"`: `List[Dict[str, Any]]` (provenance-rich)
     - `"response"`: LLM answer
     - `"n_documents_retrieved"`: chunk count
4. **ChromaDB 1.5.9 Compatibility**:
   - Replaced naive `[{} for _ in texts]` with sanitized primitive mappings and `metadatas=None` fallback.
   - Zero `ValueError: Expected metadata to be a non-empty dict` errors.

---

## 6. Architectural Transition

```
┌────────────────────────────────────────────────────────┐
│               Step 1: Naive Pipeline                  │
│ PDF -> All Pages Flatted -> Chunks -> ChromaDB [0,1,2] │
│                      (No Provenance)                   │
└──────────────────────────┬─────────────────────────────┘
                           │ Upgraded in Step 2
                           ▼
┌────────────────────────────────────────────────────────┐
│           Step 2: Structure-Aware Pipeline             │
│ PDF -> Per-Page Extraction                             │
│     -> Section Heading Matcher (Regex Heuristic)       │
│     -> Deterministic Doc ID (doc_{sha256[:12]})        │
│     -> Globally Unique Chunk IDs ({doc}_p{pg}_c{idx})  │
│     -> Rich ChromaDB Storage & Sanitized Metadatas     │
│     -> Dual Retrieval: retrieve() & retrieve_with_meta │
└────────────────────────────────────────────────────────┘
```
