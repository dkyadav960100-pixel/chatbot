"""
RAG Retriever - Main interface for the RAG system.
Combines document loading, chunking, embedding, and retrieval.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .document_loader import DocumentLoader, Document
from .chunker import TextChunker, ChunkingStrategy
from .embeddings import EmbeddingModel
from .vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG retrieval."""
    query: str
    context: str
    sources: List[Dict[str, Any]]
    num_chunks_used: int
    confidence: float  # Average similarity score
    
    def get_source_info(self) -> str:
        """Get formatted source information."""
        if not self.sources:
            return "No sources found."
        
        source_lines = []
        for i, src in enumerate(self.sources, 1):
            source_name = Path(src.get('source', 'Unknown')).name
            score = src.get('score', 0)
            source_lines.append(f"{i}. {source_name} (relevance: {score:.2f})")
        
        return "\n".join(source_lines)


class RAGRetriever:
    """
    Main RAG system that orchestrates document loading, embedding, and retrieval.
    """
    
    def __init__(
        self,
        knowledge_base_path: str,
        vector_store_path: str = "data/vector_store.db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        similarity_threshold: float = 0.3,
        device: str = "cpu"
    ):
        """
        Initialize RAG retriever.
        
        Args:
            knowledge_base_path: Path to knowledge base documents
            vector_store_path: Path to vector store database
            embedding_model: Embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval
            device: Device for embedding model
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.document_loader = DocumentLoader(str(self.knowledge_base_path))
        
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy.PARAGRAPH
        )
        
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model,
            device=device,
            cache_enabled=True
        )
        
        self.vector_store = VectorStore(
            db_path=vector_store_path,
            embedding_dim=self.embedding_model.dimension
        )
        
        self._initialized = False
        logger.info("RAG Retriever initialized")
    
    def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the RAG system by loading and indexing documents.
        
        Args:
            force_reload: Force reload even if documents are already indexed
            
        Returns:
            True if successful
        """
        try:
            # Check if already initialized
            stats = self.vector_store.get_stats()
            if stats.get('total_documents', 0) > 0 and not force_reload:
                logger.info("RAG system already initialized")
                self._initialized = True
                return True
            
            # Clear existing data if force reload
            if force_reload:
                self.vector_store.clear()
            
            # Load documents
            documents = self.document_loader.load_directory()
            
            if not documents:
                logger.warning(f"No documents found in {self.knowledge_base_path}")
                return False
            
            # Process each document
            total_chunks = 0
            
            for doc in documents:
                chunks = self.chunker.chunk_document(
                    content=doc.content,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    metadata=doc.metadata
                )
                
                if not chunks:
                    continue
                
                # Generate embeddings
                chunk_texts = [c.content for c in chunks]
                embeddings = self.embedding_model.embed_batch(
                    chunk_texts,
                    show_progress=len(chunk_texts) > 10
                )
                
                # Store in vector store
                chunk_data = [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.content,
                        chunk.source,
                        emb.embedding,
                        chunk.metadata
                    )
                    for chunk, emb in zip(chunks, embeddings)
                ]
                
                added = self.vector_store.add_chunks(chunk_data)
                total_chunks += added
                
                # Track document
                self.vector_store.add_document(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    num_chunks=len(chunks)
                )
                
                logger.info(f"Indexed document: {doc.title} ({len(chunks)} chunks)")
            
            self._initialized = True
            logger.info(f"RAG initialization complete: {len(documents)} documents, {total_chunks} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_sources: Optional[List[str]] = None
    ) -> RAGResponse:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Override default top_k
            filter_sources: Only search in specific sources
            
        Returns:
            RAGResponse with context and sources
        """
        if not self._initialized:
            self.initialize()
        
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                similarity_threshold=self.similarity_threshold
            )
            
            if not results:
                return RAGResponse(
                    query=query,
                    context="",
                    sources=[],
                    num_chunks_used=0,
                    confidence=0.0
                )
            
            # Build context from results
            context_parts = []
            sources = []
            
            for result in results:
                context_parts.append(result.content)
                sources.append({
                    "source": result.source,
                    "doc_id": result.doc_id,
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "preview": result.content[:100] + "..." if len(result.content) > 100 else result.content
                })
            
            context = "\n\n---\n\n".join(context_parts)
            avg_score = sum(r.score for r in results) / len(results)
            
            return RAGResponse(
                query=query,
                context=context,
                sources=sources,
                num_chunks_used=len(results),
                confidence=avg_score
            )
            
        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {e}")
            return RAGResponse(
                query=query,
                context="",
                sources=[],
                num_chunks_used=0,
                confidence=0.0
            )
    
    def add_document(self, file_path: str) -> bool:
        """
        Add a new document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful
        """
        try:
            doc = self.document_loader.load_file(file_path)
            
            if not doc:
                return False
            
            # Check if already exists
            if self.vector_store.document_exists(doc.doc_id):
                # Delete old version
                self.vector_store.delete_document(doc.doc_id)
            
            # Process document
            chunks = self.chunker.chunk_document(
                content=doc.content,
                doc_id=doc.doc_id,
                source=doc.source,
                metadata=doc.metadata
            )
            
            if not chunks:
                return False
            
            # Generate embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedding_model.embed_batch(chunk_texts)
            
            # Store
            chunk_data = [
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.content,
                    chunk.source,
                    emb.embedding,
                    chunk.metadata
                )
                for chunk, emb in zip(chunks, embeddings)
            ]
            
            self.vector_store.add_chunks(chunk_data)
            self.vector_store.add_document(
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                num_chunks=len(chunks)
            )
            
            logger.info(f"Added document: {doc.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        return self.vector_store.delete_document(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        store_stats = self.vector_store.get_stats()
        embedding_stats = self.embedding_model.get_cache_stats()
        
        return {
            "vector_store": store_stats,
            "embedding_cache": embedding_stats,
            "knowledge_base_path": str(self.knowledge_base_path),
            "initialized": self._initialized
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents."""
        return self.vector_store.get_all_documents()
    
    def build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a prompt for the LLM using retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            
        Returns:
            Complete prompt string
        """
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
If the context doesn't contain relevant information to answer the question, say so clearly.
Be concise but thorough in your answers."""
        
        prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def close(self):
        """Clean up resources."""
        self.vector_store.close()
        self.embedding_model.clear_cache()
