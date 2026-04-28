"""
Text chunking service for splitting documents into overlapping chunks.
"""

from typing import List


class TextChunker:
    """Service for splitting text into chunks with overlap."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: The text to split
        
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end of current chunk
            end = start + self.chunk_size
            
            # Extract chunk
            chunk = text[start:end]
            
            # Try to break at sentence/word boundary
            if end < len(text):
                # Look for the last period or newline before chunk_size
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')
                
                # Use the latest boundary found
                boundary = max(last_period, last_newline, last_space)
                
                if boundary > self.chunk_size * 0.5:  # Only break if boundary is far enough
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            chunk = chunk.strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start == end:
                start = end
        
        return chunks
    
    def chunk_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Split multiple texts into chunks.
        
        Args:
            texts: List of texts to split
        
        Returns:
            List of lists of chunks
        """
        return [self.chunk_text(text) for text in texts]


def split_text_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Convenience function to split text into chunks.
    
    Args:
        text: The text to split
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text)
