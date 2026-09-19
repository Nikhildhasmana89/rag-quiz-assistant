"""Citations and Grounded Provenance Engine for ResearchLens AI."""

from app.citations.citation_engine import (
    CitationEngine,
    STATUS_SUPPORTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE,
)

__all__ = [
    "CitationEngine",
    "STATUS_SUPPORTED",
    "STATUS_PARTIALLY_SUPPORTED",
    "STATUS_CONTRADICTED",
    "STATUS_INSUFFICIENT_EVIDENCE",
]
