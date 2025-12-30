"""
Vector store module using SQLite for persistent storage.
Stores embeddings and enables similarity search.
"""
import sqlite3
import json
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from similarity search."""
    chunk_id: str
    doc_id: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }


class VectorStore:
    """
    SQLite-based vector store for storing and searching embeddings.
    Uses numpy for similarity calculations.
    """
    
    def __init__(
        self,
        db_path: str = "vector_store.db",
        embedding_dim: int = 384
    ):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embedding vectors
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._local = threading.local()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"Initialized VectorStore at {self.db_path}")
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def _get_cursor(self):
        """Get a database cursor with automatic commit/rollback."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_cursor() as cursor:
            # Main chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Documents table for tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT,
                    num_chunks INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index for faster doc_id lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id 
                ON chunks(doc_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_source 
                ON chunks(source)
            """)
    
    def add_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        source: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a single chunk to the store.
        
        Args:
            chunk_id: Unique chunk identifier
            doc_id: Document identifier
            content: Chunk text content
            source: Document source path
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            embedding_bytes = embedding.tobytes()
            metadata_json = json.dumps(metadata or {})
            
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, doc_id, content, source, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, doc_id, content, source, embedding_bytes, metadata_json))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunk {chunk_id}: {e}")
            return False
    
    def add_chunks(
        self,
        chunks: List[Tuple[str, str, str, str, np.ndarray, Optional[Dict]]]
    ) -> int:
        """
        Add multiple chunks in batch.
        
        Args:
            chunks: List of (chunk_id, doc_id, content, source, embedding, metadata) tuples
            
        Returns:
            Number of chunks added
        """
        added = 0
        
        try:
            with self._get_cursor() as cursor:
                for chunk_id, doc_id, content, source, embedding, metadata in chunks:
                    embedding_bytes = embedding.tobytes()
                    metadata_json = json.dumps(metadata or {})
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (chunk_id, doc_id, content, source, embedding, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (chunk_id, doc_id, content, source, embedding_bytes, metadata_json))
                    added += 1
            
            logger.info(f"Added {added} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
        
        return added
    
    def add_document(
        self,
        doc_id: str,
        source: str,
        title: str,
        num_chunks: int
    ):
        """Track a document in the store."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doc_id, source, title, num_chunks)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, source, title, num_chunks))
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filter_doc_ids: Only search within these documents
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        try:
            with self._get_cursor() as cursor:
                if filter_doc_ids:
                    placeholders = ','.join(['?' for _ in filter_doc_ids])
                    cursor.execute(f"""
                        SELECT chunk_id, doc_id, content, source, embedding, metadata
                        FROM chunks
                        WHERE doc_id IN ({placeholders})
                    """, filter_doc_ids)
                else:
                    cursor.execute("""
                        SELECT chunk_id, doc_id, content, source, embedding, metadata
                        FROM chunks
                    """)
                
                rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Calculate similarities
            results = []
            
            for row in rows:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                
                # Cosine similarity (assuming normalized embeddings)
                score = float(np.dot(query_embedding, embedding))
                
                if score >= similarity_threshold:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    
                    results.append(SearchResult(
                        chunk_id=row['chunk_id'],
                        doc_id=row['doc_id'],
                        content=row['content'],
                        source=row['source'],
                        score=score,
                        metadata=metadata
                    ))
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT chunk_id, doc_id, content, source, metadata
                    FROM chunks
                    WHERE chunk_id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                
                if row:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    return SearchResult(
                        chunk_id=row['chunk_id'],
                        doc_id=row['doc_id'],
                        content=row['content'],
                        source=row['source'],
                        score=1.0,
                        metadata=metadata
                    )
                    
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
        
        return None
    
    def get_document_chunks(self, doc_id: str) -> List[SearchResult]:
        """Get all chunks for a document."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT chunk_id, doc_id, content, source, metadata
                    FROM chunks
                    WHERE doc_id = ?
                    ORDER BY chunk_id
                """, (doc_id,))
                
                rows = cursor.fetchall()
                
                return [
                    SearchResult(
                        chunk_id=row['chunk_id'],
                        doc_id=row['doc_id'],
                        content=row['content'],
                        source=row['source'],
                        score=1.0,
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error getting document chunks {doc_id}: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def clear(self):
        """Clear all data from the store."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("DELETE FROM chunks")
                cursor.execute("DELETE FROM documents")
            
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Error clearing store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM chunks")
                chunk_count = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                doc_count = cursor.fetchone()['count']
                
                return {
                    "total_chunks": chunk_count,
                    "total_documents": doc_count,
                    "embedding_dimension": self.embedding_dim,
                    "db_path": str(self.db_path)
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM documents WHERE doc_id = ?",
                    (doc_id,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get list of all indexed documents."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT doc_id, source, title, num_chunks, created_at
                    FROM documents
                    ORDER BY created_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
