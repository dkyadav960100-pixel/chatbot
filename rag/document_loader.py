"""
Document loader module for loading and processing knowledge base documents.
Supports Markdown, text, PDF, and JSON files.
"""
import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    source: str  # File path or identifier
    title: str
    metadata: Dict[str, Any]
    doc_id: str
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        import hashlib
        content_hash = hashlib.md5(
            (self.source + self.content[:100]).encode()
        ).hexdigest()[:8]
        return f"doc_{content_hash}"


class DocumentLoader:
    """
    Loads documents from various sources and formats.
    Supports: .md, .txt, .json, .pdf (with PyPDF2)
    """
    
    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.json', '.pdf'}
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize document loader.
        
        Args:
            base_path: Base directory for document loading
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._loaded_docs: Dict[str, Document] = {}
    
    def load_directory(self, directory: Optional[str] = None) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path (relative to base_path or absolute)
            
        Returns:
            List of loaded documents
        """
        dir_path = Path(directory) if directory else self.base_path
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []
        
        documents = []
        
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(str(file_path))
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {dir_path}")
        return documents
    
    def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object or None if loading fails
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            return None
        
        ext = path.suffix.lower()
        
        try:
            if ext in {'.md', '.txt'}:
                return self._load_text_file(path)
            elif ext == '.json':
                return self._load_json_file(path)
            elif ext == '.pdf':
                return self._load_pdf_file(path)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None
    
    def _load_text_file(self, path: Path) -> Document:
        """Load plain text or markdown file."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract title from content or filename
        title = self._extract_title(content, path)
        
        # Clean content
        content = self._clean_text(content)
        
        metadata = {
            'file_type': path.suffix,
            'file_size': path.stat().st_size,
            'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            'encoding': 'utf-8'
        }
        
        doc = Document(
            content=content,
            source=str(path),
            title=title,
            metadata=metadata,
            doc_id=""
        )
        
        self._loaded_docs[doc.doc_id] = doc
        return doc
    
    def _load_json_file(self, path: Path) -> Document:
        """Load JSON document file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            content = data.get('content', data.get('text', json.dumps(data, indent=2)))
            title = data.get('title', path.stem)
            metadata = data.get('metadata', {})
        elif isinstance(data, list):
            content = '\n\n'.join(
                item.get('content', item.get('text', str(item)))
                if isinstance(item, dict) else str(item)
                for item in data
            )
            title = path.stem
            metadata = {}
        else:
            content = str(data)
            title = path.stem
            metadata = {}
        
        metadata.update({
            'file_type': '.json',
            'file_size': path.stat().st_size,
        })
        
        doc = Document(
            content=content,
            source=str(path),
            title=title,
            metadata=metadata,
            doc_id=""
        )
        
        self._loaded_docs[doc.doc_id] = doc
        return doc
    
    def _load_pdf_file(self, path: Path) -> Optional[Document]:
        """Load PDF file (requires PyPDF2)."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            logger.warning("PyPDF2 not installed. Skipping PDF file.")
            return None
        
        try:
            reader = PdfReader(str(path))
            content_parts = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content_parts.append(f"[Page {page_num + 1}]\n{text}")
            
            content = '\n\n'.join(content_parts)
            
            # Try to get title from metadata or filename
            title = path.stem
            if reader.metadata and reader.metadata.title:
                title = reader.metadata.title
            
            metadata = {
                'file_type': '.pdf',
                'num_pages': len(reader.pages),
                'file_size': path.stat().st_size,
            }
            
            doc = Document(
                content=content,
                source=str(path),
                title=title,
                metadata=metadata,
                doc_id=""
            )
            
            self._loaded_docs[doc.doc_id] = doc
            return doc
            
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {e}")
            return None
    
    def _extract_title(self, content: str, path: Path) -> str:
        """Extract title from content or use filename."""
        # Try to find markdown title (# Title)
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Try to find first line as title
        first_line = content.strip().split('\n')[0].strip()
        if first_line and len(first_line) < 100:
            # Remove markdown formatting
            first_line = re.sub(r'^#+\s*', '', first_line)
            if first_line:
                return first_line
        
        # Fall back to filename
        return path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        
        return text.strip()
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a loaded document by ID."""
        return self._loaded_docs.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """Get all loaded documents."""
        return list(self._loaded_docs.values())
    
    def clear(self):
        """Clear all loaded documents."""
        self._loaded_docs.clear()


# Convenience function
def load_documents(directory: str) -> List[Document]:
    """Load all documents from a directory."""
    loader = DocumentLoader(directory)
    return loader.load_directory()
