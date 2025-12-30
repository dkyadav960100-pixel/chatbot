"""
Query cache module for caching RAG queries and responses.
Prevents re-processing of identical queries.
"""
import sqlite3
import json
import logging
import time
import hashlib
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Cache for RAG queries and LLM responses.
    Uses SQLite for persistence across restarts.
    """
    
    def __init__(
        self,
        config_or_path: Union[str, Any] = "data/cache.db",
        ttl: int = 3600,  # 1 hour default TTL
        max_size: int = 1000
    ):
        """
        Initialize query cache.
        
        Args:
            config_or_path: CacheConfig object or path to cache database
            ttl: Time-to-live for cache entries in seconds
            max_size: Maximum number of cached entries
        """
        # Handle CacheConfig object or string path
        if hasattr(config_or_path, 'cache_path'):
            # It's a CacheConfig object
            self.db_path = Path(config_or_path.cache_path)
            self.ttl = config_or_path.query_cache_ttl
            self.max_size = config_or_path.max_cache_size
        else:
            self.db_path = Path(config_or_path)
        self.ttl = ttl
        self.max_size = max_size
        self._local = threading.local()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"QueryCache initialized at {self.db_path}")
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    @contextmanager
    def _get_cursor(self):
        """Get database cursor with automatic commit."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_cursor() as cursor:
            # Query cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            
            # Embedding cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            
            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_cache_expires 
                ON query_cache(expires_at)
            """)
    
    def _generate_key(self, query: str, context_hash: Optional[str] = None) -> str:
        """Generate cache key from query and optional context."""
        key_data = query.lower().strip()
        if context_hash:
            key_data += f"_{context_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        context_hash: Optional[str] = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get cached response for a query.
        
        Args:
            query: The query string
            context_hash: Optional hash of context used
            
        Returns:
            Tuple of (response, metadata) or None if not cached
        """
        cache_key = self._generate_key(query, context_hash)
        current_time = int(time.time())
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT response, metadata, expires_at
                    FROM query_cache
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, current_time))
                
                row = cursor.fetchone()
                
                if row:
                    # Update hit count
                    cursor.execute("""
                        UPDATE query_cache
                        SET hit_count = hit_count + 1
                        WHERE cache_key = ?
                    """, (cache_key,))
                    
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    metadata['cache_hit'] = True
                    
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return row['response'], metadata
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        context_hash: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a query response.
        
        Args:
            query: The query string
            response: Response to cache
            metadata: Additional metadata
            context_hash: Hash of context used
            ttl: Custom TTL (uses default if None)
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_key(query, context_hash)
        current_time = int(time.time())
        expires_at = current_time + (ttl or self.ttl)
        
        try:
            # Enforce max size
            self._enforce_max_size()
            
            metadata_json = json.dumps(metadata or {})
            
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO query_cache
                    (cache_key, query, response, metadata, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (cache_key, query, response, metadata_json, current_time, expires_at))
            
            logger.debug(f"Cached query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def invalidate(self, query: str, context_hash: Optional[str] = None) -> bool:
        """Invalidate a specific cache entry."""
        cache_key = self._generate_key(query, context_hash)
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "DELETE FROM query_cache WHERE cache_key = ?",
                    (cache_key,)
                )
            return True
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("DELETE FROM query_cache")
                cursor.execute("DELETE FROM embedding_cache")
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        current_time = int(time.time())
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "DELETE FROM query_cache WHERE expires_at <= ?",
                    (current_time,)
                )
                deleted = cursor.rowcount
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired cache entries")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0
    
    def _enforce_max_size(self):
        """Enforce maximum cache size by removing oldest entries."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM query_cache")
                count = cursor.fetchone()['count']
                
                if count >= self.max_size:
                    # Remove oldest 10% of entries
                    to_remove = max(1, count // 10)
                    cursor.execute("""
                        DELETE FROM query_cache
                        WHERE cache_key IN (
                            SELECT cache_key FROM query_cache
                            ORDER BY created_at ASC
                            LIMIT ?
                        )
                    """, (to_remove,))
                    
                    logger.debug(f"Removed {to_remove} old cache entries")
                    
        except Exception as e:
            logger.error(f"Cache size enforcement error: {e}")
    
    def get_embedding(self, text: str, model: str) -> Optional[bytes]:
        """Get cached embedding."""
        text_hash = hashlib.md5((text + model).encode()).hexdigest()
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "SELECT embedding FROM embedding_cache WHERE text_hash = ?",
                    (text_hash,)
                )
                row = cursor.fetchone()
                return row['embedding'] if row else None
        except Exception:
            return None
    
    def set_embedding(self, text: str, model: str, embedding: bytes) -> bool:
        """Cache an embedding."""
        text_hash = hashlib.md5((text + model).encode()).hexdigest()
        current_time = int(time.time())
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO embedding_cache
                    (text_hash, embedding, model, created_at)
                    VALUES (?, ?, ?, ?)
                """, (text_hash, embedding, model, current_time))
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(hit_count) as total_hits,
                        AVG(hit_count) as avg_hits
                    FROM query_cache
                """)
                query_stats = cursor.fetchone()
                
                cursor.execute("SELECT COUNT(*) as count FROM embedding_cache")
                embedding_count = cursor.fetchone()['count']
            
            return {
                "query_cache": {
                    "entries": query_stats['total_entries'],
                    "total_hits": query_stats['total_hits'] or 0,
                    "avg_hits": round(query_stats['avg_hits'] or 0, 2)
                },
                "embedding_cache": {
                    "entries": embedding_count
                },
                "ttl": self.ttl,
                "max_size": self.max_size
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
