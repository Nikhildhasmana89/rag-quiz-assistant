"""Text chunking utilities for splitting documents."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

# Common section header pattern for academic, technical, and professional documents
SECTION_HEADING_PATTERN = re.compile(
    r"^(?:(?:[0-9IVXLCDM]+[\.\)]\s*)*)"  # Optional numbering: 1., 1.1, I., A.
    r"(abstract|introduction|background|related\s+work|literature\s+review|"
    r"system\s+architecture|proposed\s+architecture|architecture|proposed\s+methodology|methodology|proposed\s+system|approach|"
    r"implementation(?:\s+details)?|experiments?(?:\s+and\s+results)?|experimental\s+setup|"
    r"results?(?:\s+and\s+discussion)?|evaluation|empirical\s+results|discussion(?:\s+and\s+analysis)?|analysis|"
    r"conclusion(?:s)?(?:\s+and\s+future\s+work)?|summary|limitations?|references|bibliography|appendix|"
    r"technical\s+skills|skills|technologies|education|academic\s+background|qualifications|"
    r"work\s+experience|professional\s+experience|project\s+experience|experience|projects|certifications?|achievements?|publications?)"
    r"(?:\s*[:\-\u2013\u2014]?\s*.*)?$",
    re.IGNORECASE,
)


def detect_section_heading(line: str) -> Optional[str]:
    """Detect if a text line corresponds to a standard document section heading.

    Args:
        line: Single line of text

    Returns:
        Cleaned section title string if matched, otherwise None
    """
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 80:
        return None

    match = SECTION_HEADING_PATTERN.match(clean_line)
    if match:
        heading = match.group(1).strip()
        return heading.title()
    return None


class TextChunker:
    """Split text into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
    ):
        """Initialize text chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized TextChunker: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Uses a recursive character splitter that tries to preserve
        semantic boundaries (paragraphs, sentences, words).

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        from langchain_text_splitters.character import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = splitter.split_text(text)
        logger.info(
            f"Split text into {len(chunks)} chunks "
            f"(avg size: {len(text) // len(chunks) if chunks else 0} chars)"
        )

        return chunks

    def chunk_texts(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks.

        Args:
            texts: List of texts to chunk

        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        for text in texts:
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_page_with_metadata(
        self,
        page_text: str,
        page_number: int,
        document_id: str,
        source_file: str,
        document_type: str = "pdf",
        start_chunk_index: int = 0,
        initial_section: str = "",
    ) -> List[Dict[str, Any]]:
        """Chunk a single page of text while preserving structure and rich metadata.

        Args:
            page_text: Text content of the page
            page_number: 1-indexed page number
            document_id: Deterministic document identifier
            source_file: Name of source file
            document_type: Type of document (e.g. 'pdf')
            start_chunk_index: Offset for chunk indexing across the document
            initial_section: Current active section from previous page (if any)

        Returns:
            List of chunk dictionaries with full metadata
        """
        if not page_text or not page_text.strip():
            return []

        # Find headings and their character offsets in page text
        lines = page_text.splitlines()
        headings_with_offset = []
        char_offset = 0
        current_section = initial_section or ""

        for line in lines:
            heading = detect_section_heading(line)
            if heading:
                headings_with_offset.append((char_offset, heading))
            char_offset += len(line) + 1

        # Split text into chunks
        raw_chunks = self.chunk_text(page_text)
        structured_chunks = []

        # Map each chunk to the closest preceding heading
        search_start = 0
        for i, chunk in enumerate(raw_chunks):
            chunk_index = start_chunk_index + i
            chunk_id = f"{document_id}_p{page_number}_c{chunk_index}"

            prefix = chunk[:min(50, len(chunk))]
            chunk_pos = page_text.find(prefix, search_start)
            if chunk_pos != -1:
                search_start = chunk_pos
                for h_pos, h_name in headings_with_offset:
                    if h_pos <= chunk_pos + 10:
                        current_section = h_name

            structured_chunks.append({
                "text": chunk,
                "document_id": document_id,
                "source_file": source_file,
                "page_number": int(page_number),
                "section": current_section if current_section else "",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_size": len(chunk),
                "document_type": document_type,
            })

        return structured_chunks

    def chunk_with_metadata(
        self,
        text: str,
        source: str = None,
        page: int = None,
        document_id: str = None,
        section: str = None,
    ) -> List[dict]:
        """Chunk text while preserving metadata.

        Args:
            text: Text to chunk
            source: Source document identifier
            page: Page number (if applicable)
            document_id: Document identifier
            section: Section name

        Returns:
            List of dicts with chunk text and metadata
        """
        chunks = self.chunk_text(text)
        doc_id = document_id or "doc_default"
        page_num = page if page is not None else 1
        source_name = Path(source).name if source else ""
        return [
            {
                "text": chunk,
                "source": source or "",
                "source_file": source_name,
                "page": page_num,
                "page_number": page_num,
                "document_id": doc_id,
                "section": section or "",
                "chunk_id": f"{doc_id}_p{page_num}_c{idx}",
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "document_type": "pdf" if (source and str(source).endswith(".pdf")) else "text",
            }
            for idx, chunk in enumerate(chunks)
        ]
