"""
Qdrant vector database service for storing and searching document embeddings.
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from app.config import settings
import uuid


class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = None,
        vector_size: int = 768
    ):
        """
        Initialize Qdrant service.
        
        Args:
            url: Qdrant server URL. Defaults to settings.qdrant_url
            api_key: Qdrant API key. Defaults to settings.qdrant_api_key
            collection_name: Collection name. Defaults to settings.qdrant_collection_name
            vector_size: Dimension of embeddings. Defaults to 768
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
    
    def create_collection(self, recreate: bool = False) -> None:
        """
        Create collection if it doesn't exist.
        
        Args:
            recreate: If True, delete and recreate the collection
        """
        # Check if collection exists
        collections = self.client.get_collections()
        collection_exists = any(c.name == self.collection_name for c in collections.collections)
        
        if collection_exists:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
    
    def upsert_points(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Upsert points (embeddings + metadata) into the collection.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries containing:
                      - user_id: User ID
                      - project_id: Project ID
                      - document_id: Document ID
                      - filename: Original filename
                      - chunk_index: Index of chunk
                      - text: Text content
        
        Returns:
            List of point IDs
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadatas")
        
        points = []
        point_ids = []
        
        for embedding, metadata in zip(embeddings, metadatas):
            # Generate unique point ID
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            # Create point with payload
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            points.append(point)
        
        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return point_ids

    @staticmethod
    def _field_scope_condition(key: str, value: str) -> Filter | FieldCondition:
        """Match string scope values, while tolerating older numeric payloads."""
        string_value = str(value)
        if string_value.isdigit():
            return Filter(
                should=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=string_value)
                    ),
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=int(string_value))
                    )
                ]
            )
        return FieldCondition(
            key=key,
            match=MatchValue(value=string_value)
        )

    @staticmethod
    def _scope_filter(user_id: str, project_id: str) -> Filter:
        return Filter(
            must=[
                QdrantService._field_scope_condition("user_id", user_id),
                QdrantService._field_scope_condition("project_id", project_id)
            ]
        )
    
    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        project_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents with scoping filters.
        
        Args:
            query_embedding: Query embedding vector
            user_id: Filter by user ID (security)
            project_id: Filter by project ID (security)
            top_k: Number of top results to return
        
        Returns:
            List of search results with scores and payloads
        """
        # Create filter for user_id and project_id
        filter_condition = self._scope_filter(user_id, project_id)
        
        # Search. Older qdrant-client versions expose search(); newer versions
        # use query_points() for nearest-neighbor lookup.
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=top_k,
                with_payload=True
            )
        else:
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=filter_condition,
                limit=top_k,
                with_payload=True
            )
            results = query_response.points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return formatted_results

    def scroll_project_payloads(
        self,
        user_id: str,
        project_id: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Read scoped payloads for keyword/exact-ID retrieval.

        This is intentionally bounded by limit; vector search still provides the
        primary semantic candidate set.
        """
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._scope_filter(user_id, project_id),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                "id": point.id,
                "score": 0.0,
                "payload": point.payload or {}
            }
            for point in points
        ]
    
    def delete_by_document(self, document_id: int, user_id: str, project_id: str) -> None:
        """
        Delete all vectors associated with a document.
        
        Args:
            document_id: Document ID to delete
            user_id: User ID (for verification)
            project_id: Project ID (for verification)
        """
        # Create filter
        filter_condition = Filter(
            must=[
                self._field_scope_condition("user_id", user_id),
                self._field_scope_condition("project_id", project_id),
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
        
        # Delete points matching filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.name,
                "vectors_count": collection_info.points_count,
                "vectors_size": collection_info.config.params.vectors.size if collection_info.config else None
            }
        except Exception as e:
            raise RuntimeError(f"Error getting collection info: {str(e)}")


# Global Qdrant service instance
_qdrant_service = None


def get_qdrant_service(vector_size: int = 768) -> QdrantService:
    """Get or create the global Qdrant service instance."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService(vector_size=vector_size)
    return _qdrant_service
