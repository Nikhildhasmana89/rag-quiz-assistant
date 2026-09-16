"""Configuration management for RAG application.

This module handles all configuration loading from environment variables,
with support for multiple LLM providers and embedding services.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str  # "groq", "openai", or "lamini"
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""

    collection_name: str = "documents"
    persist_directory: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class PDFConfig:
    """PDF processing configuration."""

    chunk_size: int = 500
    chunk_overlap: int = 0
    extraction_method: str = "pdfplumber"  # "pdfplumber" or "pypdf"


@dataclass
class HybridConfig:
    """Hybrid retrieval (Dense + BM25) configuration."""

    enabled: bool = True
    retrieval_mode: str = "hybrid"  # "hybrid" or "vector"
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    fusion_method: str = "weighted"  # "weighted" or "rrf"
    rrf_k: int = 60
    bm25_persist_dir: str = "./data/bm25"


@dataclass
class AppConfig:
    """Main application configuration."""

    llm: LLMConfig
    embedding: EmbeddingConfig
    chroma: ChromaConfig
    pdf: PDFConfig
    hybrid: Optional[HybridConfig] = None
    input_dir: Path = Path("./data/input")
    output_dir: Path = Path("./data/output")
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Required environment variables:
    - LLM_PROVIDER: "groq", "openai", or "lamini"
    - LLM_MODEL: Model identifier (provider-specific)
    - LLM_API_KEY: API key for the provider
    - CHROMA_COLLECTION_NAME: Name for the Chroma collection
    - CHROMA_PERSIST_DIR: Directory for Chroma persistence
    - PDF_CHUNK_SIZE: Size of text chunks (default: 500)
    - PDF_CHUNK_OVERLAP: Overlap between chunks (default: 0)
    - INPUT_DIR: Directory for input files (default: ./data/input)
    - OUTPUT_DIR: Directory for output files (default: ./data/output)
    - LOG_LEVEL: Logging level (default: INFO)

    Returns:
        AppConfig: Loaded configuration

    Raises:
        ValueError: If required environment variables are missing
    """
    # Validate required variables
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in ["groq", "openai", "lamini"]:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {provider}. "
            "Must be 'groq', 'openai', or 'lamini'"
        )

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required and cannot be empty"
        )

    model = os.getenv("LLM_MODEL")
    if not model:
        # Default models per provider
        defaults = {
            "groq": "llama3-8b-8192",
            "openai": "gpt-3.5-turbo",
            "lamini": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
        model = defaults[provider]
        logger.warning(
            f"LLM_MODEL not set, using default for {provider}: {model}"
        )

    # Load path configurations
    input_dir = Path(os.getenv("INPUT_DIR", "./data/input"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "./data/output"))

    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    chroma_persist_dir = os.getenv(
        "CHROMA_PERSIST_DIR", "./data/chroma"
    )
    Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    bm25_persist_dir = os.getenv("BM25_PERSIST_DIR", "./data/bm25")
    Path(bm25_persist_dir).mkdir(parents=True, exist_ok=True)

    hybrid_enabled_str = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower()
    hybrid_enabled = hybrid_enabled_str in ("true", "1", "yes")

    hybrid_config = HybridConfig(
        enabled=hybrid_enabled,
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid").lower(),
        dense_top_k=int(os.getenv("HYBRID_DENSE_TOP_K", "20")),
        bm25_top_k=int(os.getenv("HYBRID_BM25_TOP_K", "20")),
        final_top_k=int(os.getenv("HYBRID_FINAL_TOP_K", "5")),
        dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5")),
        bm25_weight=float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")),
        fusion_method=os.getenv("HYBRID_FUSION_METHOD", "weighted").lower(),
        rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
        bm25_persist_dir=bm25_persist_dir,
    )

    # Build configuration
    config = AppConfig(
        llm=LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        ),
        embedding=EmbeddingConfig(
            model_name=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        ),
        chroma=ChromaConfig(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
            persist_directory=chroma_persist_dir,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
        ),
        pdf=PDFConfig(
            chunk_size=int(os.getenv("PDF_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("PDF_CHUNK_OVERLAP", "0")),
            extraction_method=os.getenv(
                "PDF_EXTRACTION_METHOD", "pdfplumber"
            ),
        ),
        hybrid=hybrid_config,
        input_dir=input_dir,
        output_dir=output_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    return config


def validate_config(config: AppConfig) -> bool:
    """Validate configuration settings.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If validation fails
    """
    if config.pdf.chunk_size <= 0:
        raise ValueError("PDF chunk_size must be positive")

    if config.pdf.chunk_overlap < 0:
        raise ValueError("PDF chunk_overlap cannot be negative")

    if config.pdf.chunk_overlap >= config.pdf.chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if config.embedding.batch_size <= 0:
        raise ValueError("Embedding batch_size must be positive")

    if config.llm.temperature < 0 or config.llm.temperature > 2:
        raise ValueError("LLM temperature must be between 0 and 2")

    if config.llm.max_tokens <= 0:
        raise ValueError("LLM max_tokens must be positive")

    if config.hybrid:
        if config.hybrid.dense_top_k <= 0:
            raise ValueError("Hybrid dense_top_k must be positive")
        if config.hybrid.bm25_top_k <= 0:
            raise ValueError("Hybrid bm25_top_k must be positive")
        if config.hybrid.final_top_k <= 0:
            raise ValueError("Hybrid final_top_k must be positive")
        if config.hybrid.dense_weight < 0:
            raise ValueError("Hybrid dense_weight cannot be negative")
        if config.hybrid.bm25_weight < 0:
            raise ValueError("Hybrid bm25_weight cannot be negative")
        if config.hybrid.fusion_method not in ["weighted", "rrf"]:
            raise ValueError(
                f"Invalid fusion_method: {config.hybrid.fusion_method}. "
                "Must be 'weighted' or 'rrf'"
            )
        if config.hybrid.retrieval_mode not in ["hybrid", "vector"]:
            raise ValueError(
                f"Invalid retrieval_mode: {config.hybrid.retrieval_mode}. "
                "Must be 'hybrid' or 'vector'"
            )
        if config.hybrid.rrf_k <= 0:
            raise ValueError("Hybrid rrf_k must be positive")

    logger.info("Configuration validation passed")
    return True
