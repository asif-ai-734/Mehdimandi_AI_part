"""
Embeddings service using Sentence Transformers.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingsService:
    """Service for generating embeddings using Sentence Transformers."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embeddings service.
        
        Args:
            model_name: Name of the Sentence Transformer model to use.
                       Defaults to settings.embedding_model_name
        """
        self.model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize the embedding
        
        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embeddings
            batch_size: Batch size for processing. Defaults to settings.embedding_batch_size
        
        Returns:
            List of embeddings
        """
        if batch_size is None:
            batch_size = settings.embedding_batch_size
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
    
    @staticmethod
    def normalize_embedding(embedding: Union[List[float], np.ndarray]) -> List[float]:
        """
        Normalize an embedding vector.
        
        Args:
            embedding: Embedding vector to normalize
        
        Returns:
            Normalized embedding
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.tolist()
        
        normalized = embedding / norm
        return normalized.tolist()


# Global embeddings service instance
_embeddings_service = None


def get_embeddings_service() -> EmbeddingsService:
    """Get or create the global embeddings service instance."""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service


def embed_query(query: str) -> List[float]:
    """Convenience function to embed a single query."""
    service = get_embeddings_service()
    return service.embed_text(query, normalize=True)


def embed_documents(documents: List[str]) -> List[List[float]]:
    """Convenience function to embed multiple documents."""
    service = get_embeddings_service()
    return service.embed_texts(documents, normalize=True)
