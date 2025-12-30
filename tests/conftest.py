# Test configuration
import pytest

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "title": "Test Document 1",
            "content": "This is test content for document one. It contains important information.",
            "source": "test1.md"
        },
        {
            "title": "Test Document 2", 
            "content": "Second document with different content. More details here.",
            "source": "test2.md"
        }
    ]

@pytest.fixture
def sample_image_bytes():
    """Provide sample image bytes for testing."""
    from PIL import Image
    import io
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()
