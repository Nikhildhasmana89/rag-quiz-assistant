"""Vector store and embedding management using ChromaDB."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using ChromaDB for embedding storage and retrieval."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence
            embedding_model: Name of embedding model
        """
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Create embedding function
        self.embedding_function = (
            SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

        logger.info(
            f"Initialized ChromaVectorStore: "
            f"collection={collection_name}, model={embedding_model}"
        )

    def create_collection(self, force_recreate: bool = False):
        """Create or get collection.

        Args:
            force_recreate: If True, delete existing collection and recreate
        """
        if force_recreate:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = None
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"Collection did not exist or could not be deleted: {e}")

        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            # Collection may already exist
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Using existing collection: {self.collection_name}")

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        """Add documents to the vector store.

        Args:
            texts: List of document texts
            ids: Optional list of document IDs
            metadata: Optional list of metadata dicts

        Raises:
            ValueError: If collection not created
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first")

        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        metadatas_to_add = None
        if metadata is not None:
            sanitized = []
            for m in metadata:
                if not m:
                    sanitized.append({"indexed": True})
                    continue
                clean_m = {}
                for k, v in m.items():
                    if v is None:
                        clean_m[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        clean_m[k] = v
                    else:
                        clean_m[k] = str(v)
                sanitized.append(clean_m if clean_m else {"indexed": True})
            metadatas_to_add = sanitized

        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas_to_add,
            )
            logger.info(f"Added {len(texts)} documents to collection")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Query the vector store.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return

        Returns:
            Query results with documents, metadatas, and distances
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first")

        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
            )
            logger.info(f"Query returned {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            raise

    def sample_documents(self, n: int = 8) -> List[str]:
        """Sample up to n chunks from the collection (for quiz generation).

        Args:
            n: Maximum number of chunks to sample

        Returns:
            List of document chunk texts (randomly sampled if more than n exist)
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first")

        import random

        try:
            all_data = self.collection.get()
            documents = all_data.get("documents") or []
            if len(documents) <= n:
                return documents
            return random.sample(documents, n)
        except Exception as e:
            logger.error(f"Error sampling documents: {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.

        Returns:
            Collection metadata and count
        """
        if self.collection is None:
            raise ValueError("Collection not created")

        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise

    def delete_collection(self):
        """Delete the current collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.collection = None
            logger.debug(f"Could not delete collection: {e}")

    def clear_collection(self):
        """Clear all documents from the collection while keeping it."""
        if self.collection is None:
            raise ValueError("Collection not created")

        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
