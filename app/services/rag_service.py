"""
RAG (Retrieval-Augmented Generation) orchestration service.
Coordinates file extraction, chunking, embedding, and retrieval.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.models import Document, ChatHistory
from app.services.file_extractor import extract_text
from app.services.chunker import split_text_into_chunks
from app.services.embeddings import get_embeddings_service
from app.services.qdrant_service import get_qdrant_service
from app.services.openai_service import get_openai_service
from app.config import settings
import json


QUERY_ID_PATTERN = re.compile(
    r"\b(?:[A-Z]{1,3}-\d{2,4}[A-Z]?|\d{1,2}\.\d{2}|[WD]\d{2,4}[A-Z]?|"
    r"[A-Z]{1,3}-\d{2,4}/\d+[A-Z]?|\d+[A-Z]?/[A-Z]{1,3}-\d{2,4})\b",
    re.IGNORECASE,
)
QUERY_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9./-]{2,}")
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "where",
    "when", "which", "are", "does", "show", "sheet", "page", "document",
}


@dataclass
class DocumentChunk:
    """Text plus metadata ready for embedding."""

    text: str
    metadata: Dict[str, Any]


class RAGService:
    """Orchestration service for the RAG system."""
    
    def __init__(self):
        """Initialize RAG service with dependencies."""
        self.embeddings_service = get_embeddings_service()
        self.qdrant_service = get_qdrant_service(
            vector_size=self.embeddings_service.get_embedding_dimension()
        )
        self.openai_service = get_openai_service()
    
    def process_document(
        self,
        file_path: str,
        file_type: str,
        document_id: int,
        user_id: str,
        project_id: str,
        project_name: str,
        project_address: str,
        filename: str,
        db: Session,
        source_type: str = "document",
    ) -> Tuple[int, str]:
        """
        Process a single document: extract text, chunk, embed, and store.
        
        Args:
            file_path: Path to the uploaded file
            file_type: File type (pdf, docx, txt)
            document_id: Database document ID
            user_id: User ID
            project_id: Project ID
            filename: Original filename
            db: Database session
        
        Returns:
            Tuple of (total_chunks, error_message or empty string)
        """
        try:
            # Extract text
            text = extract_text(file_path, file_type)
            if not text or len(text.strip()) == 0:
                return 0, "No text content extracted from file"
            
            # Split into chunks
            chunks = split_text_into_chunks(
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            if not chunks:
                return 0, "Text could not be split into chunks"

            text_chunks = [
                DocumentChunk(
                    text=chunk,
                    metadata={
                        "chunk_type": "raw_text",
                        "source_text_ref": filename,
                        "source_type": source_type,
                        "project_name": project_name,
                        "project_address": project_address,
                        "entities": [],
                    },
                )
                for chunk in chunks
            ]
            return self._embed_and_store_chunks(
                chunks=text_chunks,
                document_id=document_id,
                user_id=user_id,
                project_id=project_id,
                project_name=project_name,
                project_address=project_address,
                filename=filename,
                source_type=source_type,
                db=db,
            )
        
        except Exception as e:
            return 0, f"Error processing document: {str(e)}"

    def _embed_and_store_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: int,
        user_id: str,
        project_id: str,
        project_name: str,
        project_address: str,
        filename: str,
        source_type: str,
        db: Session,
    ) -> Tuple[int, str]:
        """Embed chunks, store payload metadata, and update the document row."""
        chunk_texts = [chunk.text for chunk in chunks if chunk.text.strip()]
        if not chunk_texts:
            return 0, "No non-empty chunks generated"

        embeddings = self.embeddings_service.embed_texts(chunk_texts, normalize=True)
        metadatas = []
        chunk_index = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "user_id": user_id,
                    "project_id": project_id,
                    "project_name": project_name,
                    "project_address": project_address,
                    "document_id": document_id,
                    "filename": filename,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                    "text": chunk.text,
                }
            )
            metadatas.append(metadata)
            chunk_index += 1

        self.qdrant_service.upsert_points(embeddings, metadatas)

        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.total_chunks = len(metadatas)
            db.commit()

        return len(metadatas), ""
    
    def retrieve_context(
        self,
        query: str,
        user_id: str,
        project_id: str,
        top_k: int = None,
        db: Session = None
    ) -> Tuple[str, List[str]]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query: User's question
            user_id: User ID for filtering
            project_id: Project ID for filtering
            top_k: Number of top results. Defaults to settings.top_k_documents
            db: Database session (optional, for document lookup)
        
        Returns:
            Tuple of (context_string, source_filenames)
        """
        if top_k is None:
            top_k = settings.top_k_documents
        
        # Embed query
        query_embedding = self.embeddings_service.embed_text(query, normalize=True)

        candidate_limit = max(
            top_k,
            top_k * max(settings.hybrid_candidate_multiplier, 1),
        )

        vector_results = self.qdrant_service.search(
            query_embedding=query_embedding,
            user_id=user_id,
            project_id=project_id,
            top_k=candidate_limit
        )

        keyword_candidates = []
        try:
            keyword_candidates = self.qdrant_service.scroll_project_payloads(
                user_id=user_id,
                project_id=project_id,
                limit=settings.keyword_scan_limit,
            )
        except Exception:
            keyword_candidates = []

        results = self._rerank_hybrid_results(
            query=query,
            vector_results=vector_results,
            keyword_candidates=keyword_candidates,
            top_k=top_k,
        )

        # Extract context and sources
        context_parts = []
        sources = []

        for index, result in enumerate(results, start=1):
            payload = result["payload"]
            text = payload.get("text", "")

            if text:
                source = self._format_source(payload, index)
                context_parts.append(f"[S{index}] {source}\n{text}\n")
                sources.append(source)

        context = "\n".join(context_parts)

        return context, sources

    def _rerank_hybrid_results(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        keyword_candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse semantic vector results with exact-ID/keyword hits."""
        combined: Dict[str, Dict[str, Any]] = {}

        for result in vector_results:
            point_id = str(result["id"])
            combined[point_id] = {
                **result,
                "vector_score": float(result.get("score", 0.0) or 0.0),
                "keyword_score": 0.0,
            }

        keyword_stats = self._keyword_statistics(query, keyword_candidates)
        for candidate in keyword_candidates:
            keyword_score = self._keyword_score(
                query,
                candidate.get("payload", {}),
                keyword_stats,
            )
            if keyword_score <= 0:
                continue
            point_id = str(candidate["id"])
            if point_id not in combined:
                combined[point_id] = {
                    **candidate,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                }
            else:
                combined[point_id]["keyword_score"] = max(
                    combined[point_id].get("keyword_score", 0.0),
                    keyword_score,
                )

        reranked = []
        for result in combined.values():
            result["score"] = (
                result.get("vector_score", 0.0)
                + (0.35 * result.get("keyword_score", 0.0))
            )
            reranked.append(result)

        return sorted(reranked, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]

    def _keyword_score(
        self,
        query: str,
        payload: Dict[str, Any],
        keyword_stats: Dict[str, Any] = None,
    ) -> float:
        haystack = self._payload_search_text(payload).lower()
        if not haystack:
            return 0.0

        query_ids = [match.upper() for match in QUERY_ID_PATTERN.findall(query)]
        entity_values = {
            str(entity).upper()
            for entity in payload.get("entities", [])
            if str(entity).strip()
        }
        exact_hits = sum(1 for query_id in query_ids if query_id.lower() in haystack)
        entity_hits = sum(1 for query_id in query_ids if query_id in entity_values)

        bm25_score = 0.0
        if keyword_stats:
            bm25_score = self._bm25_score(haystack, keyword_stats)

        return (5.0 * exact_hits) + (3.0 * entity_hits) + bm25_score

    def _keyword_statistics(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        query_terms = [
            token.lower()
            for token in QUERY_TOKEN_PATTERN.findall(query)
            if token.lower() not in STOPWORDS
        ]
        query_terms = sorted(set(query_terms))
        if not query_terms or not candidates:
            return {"query_terms": [], "avgdl": 1.0, "idf": {}}

        document_terms = []
        for candidate in candidates:
            text = self._payload_search_text(candidate.get("payload", {})).lower()
            terms = QUERY_TOKEN_PATTERN.findall(text)
            document_terms.append(terms)

        document_count = max(len(document_terms), 1)
        avgdl = sum(len(terms) for terms in document_terms) / document_count
        avgdl = max(avgdl, 1.0)

        idf = {}
        for term in query_terms:
            document_frequency = sum(1 for terms in document_terms if term in set(terms))
            idf[term] = math.log(1 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)))

        return {
            "query_terms": query_terms,
            "avgdl": avgdl,
            "idf": idf,
        }

    @staticmethod
    def _bm25_score(haystack: str, keyword_stats: Dict[str, Any]) -> float:
        terms = QUERY_TOKEN_PATTERN.findall(haystack)
        if not terms:
            return 0.0

        term_counts = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        k1 = 1.5
        b = 0.75
        document_length = len(terms)
        avgdl = keyword_stats.get("avgdl", 1.0) or 1.0
        for term in keyword_stats.get("query_terms", []):
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue
            idf = keyword_stats.get("idf", {}).get(term, 0.0)
            denominator = frequency + k1 * (1 - b + b * (document_length / avgdl))
            score += idf * ((frequency * (k1 + 1)) / denominator)
        return score

    @staticmethod
    def _payload_search_text(payload: Dict[str, Any]) -> str:
        fields = [
            payload.get("text", ""),
            payload.get("filename", ""),
            payload.get("source_type", ""),
            payload.get("project_name", ""),
            payload.get("project_address", ""),
            payload.get("sheet", ""),
            payload.get("chunk_type", ""),
            " ".join(str(entity) for entity in payload.get("entities", [])),
        ]
        return " ".join(str(field) for field in fields if field)

    @staticmethod
    def _format_source(payload: Dict[str, Any], index: int) -> str:
        parts = [f"S{index}: {payload.get('filename', 'unknown')}"]
        if payload.get("source_type"):
            parts.append(payload["source_type"])
        if payload.get("project_name"):
            parts.append(f"project {payload['project_name']}")
        if payload.get("project_address"):
            parts.append(f"address {payload['project_address']}")
        if payload.get("sheet"):
            parts.append(f"sheet {payload['sheet']}")
        if payload.get("page"):
            parts.append(f"page {payload['page']}")
        if payload.get("chunk_type"):
            parts.append(payload["chunk_type"])
        if payload.get("source_text_ref"):
            parts.append(f"text {payload['source_text_ref']}")
        return " | ".join(parts)
    
    def generate_chat_response(
        self,
        query: str,
        user_id: str,
        project_id: str,
        db: Session
    ) -> Tuple[str, List[str]]:
        """
        Generate a chat response using RAG.
        
        Args:
            query: User's question
            user_id: User ID
            project_id: Project ID
            db: Database session
        
        Returns:
            Tuple of (response, source_filenames)
        """
        # Retrieve context
        context, sources = self.retrieve_context(
            query=query,
            user_id=user_id,
            project_id=project_id,
            db=db
        )
        
        if not context.strip():
            response = "I could not find any relevant documents in your project. Please upload documents to ask questions."
            sources = []
        else:
            # Generate response using OpenAI
            response = self.openai_service.generate_response(
                query=query,
                context=context,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract any sources mentioned in response
            sources = self.openai_service.extract_sources_from_response(
                response, sources
            )
        
        return response, sources
    
    def save_chat_history(
        self,
        user_id: str,
        project_id: str,
        user_message: str,
        assistant_response: str,
        sources: List[str],
        db: Session
    ) -> ChatHistory:
        """
        Save chat message to history.
        
        Args:
            user_id: User ID
            project_id: Project ID
            user_message: User's message
            assistant_response: Assistant's response
            sources: List of source filenames
            db: Database session
        
        Returns:
            ChatHistory record
        """
        # Convert sources to JSON string
        sources_json = json.dumps(sources) if sources else None
        
        # Create chat history record
        chat = ChatHistory(
            user_id=user_id,
            project_id=project_id,
            user_message=user_message,
            assistant_response=assistant_response,
            sources=sources_json
        )
        
        db.add(chat)
        db.commit()
        db.refresh(chat)
        
        return chat


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
