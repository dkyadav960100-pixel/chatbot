"""
Text chunking module for splitting documents into manageable pieces.
Implements multiple chunking strategies with overlap support.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Generator, Union, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"  # Based on markdown headers


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    content: str
    chunk_id: str
    doc_id: str
    source: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    metadata: dict
    
    @property
    def preview(self) -> str:
        """Get a short preview of the chunk."""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content


class TextChunker:
    """
    Splits documents into chunks for embedding and retrieval.
    Supports multiple strategies with configurable overlap.
    """
    
    # Sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    # Paragraph pattern (double newline)
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
    
    # Markdown header pattern
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(
        self,
        config_or_size: Union[int, Any] = 500,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        min_chunk_size: int = 50
    ):
        """
        Initialize the text chunker.
        
        Args:
            config_or_size: RAGConfig object or target size for each chunk (in characters)
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy to use
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        # Handle RAGConfig object or int
        if hasattr(config_or_size, 'chunk_size'):
            # It's a RAGConfig object
            self.chunk_size = config_or_size.chunk_size
            self.chunk_overlap = config_or_size.chunk_overlap
            self.strategy = strategy
            self.min_chunk_size = min_chunk_size
        else:
            self.chunk_size = config_or_size
            self.chunk_overlap = chunk_overlap
            self.strategy = strategy
            self.min_chunk_size = min_chunk_size
        
        # Validate parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_document(
        self,
        content: str,
        doc_id: str,
        source: str,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Split document content into chunks.
        
        Args:
            content: Document text content
            doc_id: Document identifier
            source: Document source path
            metadata: Additional metadata to include
            
        Returns:
            List of TextChunk objects
        """
        if not content or not content.strip():
            logger.warning(f"Empty content for document {doc_id}")
            return []
        
        metadata = metadata or {}
        
        # Choose chunking method based on strategy
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            raw_chunks = self._chunk_fixed_size(content)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            raw_chunks = self._chunk_by_sentence(content)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            raw_chunks = self._chunk_by_paragraph(content)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self._chunk_semantic(content)
        else:
            raw_chunks = self._chunk_fixed_size(content)
        
        # Convert to TextChunk objects
        chunks = []
        total_chunks = len(raw_chunks)
        
        for idx, (text, start, end) in enumerate(raw_chunks):
            chunk = TextChunk(
                content=text,
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                source=source,
                start_char=start,
                end_char=end,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata={
                    **metadata,
                    'chunk_strategy': self.strategy.value,
                    'chunk_size_target': self.chunk_size,
                }
            )
            chunks.append(chunk)
        
        logger.debug(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks
    
    def _chunk_fixed_size(self, text: str) -> List[tuple]:
        """
        Split text into fixed-size chunks with overlap.
        Tries to break at word boundaries.
        
        Returns:
            List of (content, start_char, end_char) tuples
        """
        chunks = []
        text_length = len(text)
        
        if text_length <= self.chunk_size:
            return [(text.strip(), 0, text_length)]
        
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Try to find a good break point (word boundary)
            if end < text_length:
                # Look for space, newline, or punctuation
                break_point = self._find_break_point(text, start, end)
                if break_point > start:
                    end = break_point
            
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, start, end))
            elif chunk_text and chunks:
                # Merge small chunk with previous
                prev_text, prev_start, _ = chunks[-1]
                chunks[-1] = (prev_text + " " + chunk_text, prev_start, end)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= chunks[-1][1] if chunks else 0:
                start = end  # Prevent infinite loop
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[tuple]:
        """
        Split text by sentences, grouping until chunk_size is reached.
        """
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [(text.strip(), 0, len(text))]
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, chunk_start, chunk_end))
                
                # Start new chunk with overlap (last sentence)
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_length
                    chunk_start = chunk_end - len(overlap_text)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
                    chunk_start = chunk_end
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[tuple]:
        """
        Split text by paragraphs, grouping small paragraphs together.
        """
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return [(text.strip(), 0, len(text))]
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, chunk_start, chunk_end))
                
                # Start new chunk
                current_chunk = [para]
                current_length = para_length
                chunk_start = chunk_end + 2  # Account for paragraph separator
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # Account for separator
        
        # Last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[tuple]:
        """
        Split text by semantic boundaries (markdown headers).
        Falls back to paragraph chunking if no headers found.
        """
        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(text))
        
        if not headers:
            return self._chunk_by_paragraph(text)
        
        chunks = []
        
        for i, match in enumerate(headers):
            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            
            section_text = text[start:end].strip()
            
            # If section is too large, sub-chunk it
            if len(section_text) > self.chunk_size * 2:
                sub_chunks = self._chunk_fixed_size(section_text)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section_text) >= self.min_chunk_size:
                chunks.append((section_text, start, end))
        
        # Handle content before first header
        if headers and headers[0].start() > 0:
            pre_content = text[:headers[0].start()].strip()
            if len(pre_content) >= self.min_chunk_size:
                chunks.insert(0, (pre_content, 0, headers[0].start()))
        
        return chunks
    
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point near the end position."""
        # Look backwards from end for a break point
        search_region = text[start:end]
        
        # Priority: newline > period > space
        for char in ['\n', '.', ' ', ',']:
            last_pos = search_region.rfind(char)
            if last_pos > len(search_region) * 0.5:  # At least halfway through
                return start + last_pos + 1
        
        return end
    
    def chunk_documents(
        self,
        documents: List['Document']
    ) -> Generator[TextChunk, None, None]:
        """
        Chunk multiple documents, yielding chunks one at a time.
        
        Args:
            documents: List of Document objects
            
        Yields:
            TextChunk objects
        """
        for doc in documents:
            chunks = self.chunk_document(
                content=doc.content,
                doc_id=doc.doc_id,
                source=doc.source,
                metadata=doc.metadata
            )
            for chunk in chunks:
                yield chunk


def create_chunks(
    content: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[TextChunk]:
    """Convenience function to chunk text."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(content, "doc_0", "inline")
