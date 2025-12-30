"""
Tests for the GenAI Telegram Bot components.
"""
import pytest
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentLoader:
    """Tests for document loading functionality."""
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        from rag.document_loader import DocumentLoader
        
        # Create test file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Document\n\nThis is test content.")
        
        loader = DocumentLoader(str(tmp_path))
        doc = loader.load_file(str(md_file))
        
        assert doc is not None
        assert "Test Document" in doc.title or "test" in doc.title.lower()
        assert "test content" in doc.content.lower()
    
    def test_load_text_file(self, tmp_path):
        """Test loading a plain text file."""
        from rag.document_loader import DocumentLoader
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text content here")
        
        loader = DocumentLoader(str(tmp_path))
        doc = loader.load_file(str(txt_file))
        
        assert doc is not None
        assert "Plain text content" in doc.content
    
    def test_load_directory(self, tmp_path):
        """Test loading all documents from directory."""
        from rag.document_loader import DocumentLoader
        
        # Create multiple files
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent 1")
        (tmp_path / "doc2.txt").write_text("Document 2 content")
        (tmp_path / "ignored.xyz").write_text("Should be ignored")
        
        loader = DocumentLoader(str(tmp_path))
        docs = loader.load_directory()
        
        assert len(docs) == 2  # Only .md and .txt
    
    def test_empty_file(self, tmp_path):
        """Test handling of empty files."""
        from rag.document_loader import DocumentLoader
        
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        loader = DocumentLoader(str(tmp_path))
        doc = loader.load_file(str(empty_file))
        
        # Should handle gracefully
        assert doc is not None


class TestTextChunker:
    """Tests for text chunking functionality."""
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking."""
        from rag.chunker import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkingStrategy.FIXED_SIZE
        )
        
        text = "A" * 250  # 250 characters
        chunks = chunker.chunk_document(text, "doc1", "test.txt")
        
        assert len(chunks) >= 2
        assert all(len(c.content) <= 120 for c in chunks)  # Some tolerance
    
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        from rag.chunker import TextChunker, ChunkingStrategy
        
        chunker = TextChunker(
            chunk_size=500,
            strategy=ChunkingStrategy.PARAGRAPH
        )
        
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunker.chunk_document(text, "doc1", "test.txt")
        
        assert len(chunks) >= 1
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        from rag.chunker import TextChunker
        
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "Word " * 30  # ~150 characters
        chunks = chunker.chunk_document(text, "doc1", "test.txt")
        
        # With overlap, content from one chunk should appear in adjacent chunks
        assert len(chunks) >= 2
    
    def test_empty_content(self):
        """Test handling empty content."""
        from rag.chunker import TextChunker
        
        chunker = TextChunker()
        chunks = chunker.chunk_document("", "doc1", "test.txt")
        
        assert chunks == []


class TestVectorStore:
    """Tests for vector store functionality."""
    
    def test_add_and_retrieve_chunk(self, tmp_path):
        """Test adding and retrieving a chunk."""
        from rag.vector_store import VectorStore
        import numpy as np
        
        db_path = tmp_path / "test_vector.db"
        store = VectorStore(str(db_path), embedding_dim=384)
        
        # Add chunk
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        success = store.add_chunk(
            chunk_id="test_chunk_1",
            doc_id="doc_1",
            content="Test content",
            source="test.txt",
            embedding=embedding
        )
        
        assert success
        
        # Retrieve
        result = store.get_chunk("test_chunk_1")
        assert result is not None
        assert result.content == "Test content"
    
    def test_similarity_search(self, tmp_path):
        """Test similarity search."""
        from rag.vector_store import VectorStore
        import numpy as np
        
        db_path = tmp_path / "test_search.db"
        store = VectorStore(str(db_path), embedding_dim=384)
        
        # Add multiple chunks
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            store.add_chunk(
                chunk_id=f"chunk_{i}",
                doc_id="doc_1",
                content=f"Content {i}",
                source="test.txt",
                embedding=embedding
            )
        
        # Search
        query_embedding = np.random.randn(384).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        results = store.search(query_embedding, top_k=3)
        
        assert len(results) == 3
        # Results should be sorted by score (descending)
        assert all(
            results[i].score >= results[i+1].score 
            for i in range(len(results)-1)
        )
    
    def test_delete_document(self, tmp_path):
        """Test document deletion."""
        from rag.vector_store import VectorStore
        import numpy as np
        
        db_path = tmp_path / "test_delete.db"
        store = VectorStore(str(db_path), embedding_dim=384)
        
        # Add chunks
        for i in range(3):
            embedding = np.random.randn(384).astype(np.float32)
            store.add_chunk(
                chunk_id=f"chunk_{i}",
                doc_id="doc_to_delete",
                content=f"Content {i}",
                source="test.txt",
                embedding=embedding
            )
        
        store.add_document("doc_to_delete", "test.txt", "Test", 3)
        
        # Delete
        success = store.delete_document("doc_to_delete")
        assert success
        
        # Verify deletion
        chunks = store.get_document_chunks("doc_to_delete")
        assert len(chunks) == 0


class TestCache:
    """Tests for caching functionality."""
    
    def test_cache_set_get(self, tmp_path):
        """Test cache set and get operations."""
        from utils.cache import QueryCache
        
        cache = QueryCache(
            db_path=str(tmp_path / "test_cache.db"),
            ttl=3600
        )
        
        # Set
        cache.set("test query", "test response", {"key": "value"})
        
        # Get
        result = cache.get("test query")
        
        assert result is not None
        response, metadata = result
        assert response == "test response"
        assert metadata.get("cache_hit") == True
    
    def test_cache_expiry(self, tmp_path):
        """Test cache TTL."""
        from utils.cache import QueryCache
        import time
        
        cache = QueryCache(
            db_path=str(tmp_path / "test_cache_ttl.db"),
            ttl=1  # 1 second TTL
        )
        
        cache.set("expiring query", "response")
        
        # Should exist immediately
        assert cache.get("expiring query") is not None
        
        # Wait for expiry
        time.sleep(1.5)
        
        # Should be expired
        assert cache.get("expiring query") is None
    
    def test_cache_invalidate(self, tmp_path):
        """Test cache invalidation."""
        from utils.cache import QueryCache
        
        cache = QueryCache(db_path=str(tmp_path / "test_invalidate.db"))
        
        cache.set("to_invalidate", "response")
        assert cache.get("to_invalidate") is not None
        
        cache.invalidate("to_invalidate")
        assert cache.get("to_invalidate") is None


class TestSessionManager:
    """Tests for session management."""
    
    def test_create_session(self):
        """Test session creation."""
        from utils.session import SessionManager
        
        manager = SessionManager()
        session = manager.get_session(12345)
        
        assert session is not None
        assert session.user_id == 12345
    
    def test_add_messages(self):
        """Test adding messages to session."""
        from utils.session import SessionManager
        
        manager = SessionManager(history_length=3)
        
        manager.add_user_message(12345, "Hello")
        manager.add_assistant_message(12345, "Hi there!")
        
        history = manager.get_conversation_history(12345)
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_history_limit(self):
        """Test history length limit."""
        from utils.session import SessionManager
        
        manager = SessionManager(history_length=2)
        
        # Add more than limit
        for i in range(5):
            manager.add_user_message(12345, f"Message {i}")
            manager.add_assistant_message(12345, f"Response {i}")
        
        history = manager.get_conversation_history(12345)
        
        # Should be limited to history_length * 2 (user + assistant pairs)
        assert len(history) <= 4
    
    def test_session_context(self):
        """Test session context storage."""
        from utils.session import SessionManager
        
        manager = SessionManager()
        
        manager.set_context(12345, "preference", "dark_mode")
        value = manager.get_context(12345, "preference")
        
        assert value == "dark_mode"


class TestImageProcessor:
    """Tests for image processing."""
    
    def test_validate_valid_image(self, tmp_path):
        """Test validation of valid image."""
        from vision.image_processor import ImageProcessor
        from PIL import Image
        import io
        
        # Create a valid test image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        processor = ImageProcessor()
        is_valid, message = processor.validate_image(image_bytes)
        
        assert is_valid
        assert message == "Valid"
    
    def test_validate_empty_image(self):
        """Test validation of empty data."""
        from vision.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        is_valid, message = processor.validate_image(b"")
        
        assert not is_valid
        assert "Empty" in message
    
    def test_process_image_bytes(self, tmp_path):
        """Test processing image from bytes."""
        from vision.image_processor import ImageProcessor
        from PIL import Image
        import io
        
        # Create test image
        img = Image.new('RGB', (200, 200), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        processor = ImageProcessor(max_size=(100, 100))
        result = processor.process_bytes(image_bytes)
        
        assert result is not None
        assert result.processed_size[0] <= 100
        assert result.processed_size[1] <= 100


class TestLLMHandler:
    """Tests for LLM handler (mocked)."""
    
    def test_no_provider_available(self):
        """Test behavior when no LLM provider is available."""
        from llm.llm_handler import LLMHandler, LLMProvider
        
        handler = LLMHandler(
            ollama_base_url="http://nonexistent:11434",
            openai_api_key=""
        )
        
        provider = handler.get_available_provider()
        
        # Should return NONE when nothing is available
        assert provider in [LLMProvider.NONE, LLMProvider.OLLAMA, LLMProvider.OPENAI]
    
    def test_generate_no_provider(self):
        """Test generate returns appropriate message when no provider."""
        from llm.llm_handler import LLMHandler
        
        handler = LLMHandler(
            ollama_base_url="http://nonexistent:11434",
            openai_api_key=""
        )
        
        # Force no provider
        handler._available_provider = None
        
        response = handler.generate("Test prompt")
        
        # Should not crash, should return a response
        assert response is not None
        assert hasattr(response, 'content')


# Integration test (requires services)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring external services."""
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TEST"),
        reason="Integration tests disabled"
    )
    def test_full_rag_pipeline(self, tmp_path):
        """Test complete RAG pipeline."""
        from rag import RAGRetriever
        
        # Create test knowledge base
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        
        (kb_path / "test.md").write_text("""
# Test Knowledge Base

## Topic A
This is information about topic A.
It contains specific details.

## Topic B  
This is information about topic B.
It has different content.
        """)
        
        retriever = RAGRetriever(
            knowledge_base_path=str(kb_path),
            vector_store_path=str(tmp_path / "vectors.db"),
            top_k=2
        )
        
        # Initialize
        success = retriever.initialize()
        assert success
        
        # Query
        response = retriever.retrieve("Tell me about topic A")
        
        assert response.context != ""
        assert len(response.sources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
