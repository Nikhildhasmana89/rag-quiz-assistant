"""Evidence verification and hallucination detection module."""

from app.verification.evidence_verifier import (
    EvidenceVerifier,
    STATUS_SUPPORTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE,
    VALID_VERIFICATION_STATUSES,
)

__all__ = [
    "EvidenceVerifier",
    "STATUS_SUPPORTED",
    "STATUS_PARTIALLY_SUPPORTED",
    "STATUS_CONTRADICTED",
    "STATUS_INSUFFICIENT_EVIDENCE",
    "VALID_VERIFICATION_STATUSES",
]
