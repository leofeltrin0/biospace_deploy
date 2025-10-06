"""
Tests for chunking module
"""

import pytest
from modules.chunking import TextChunker


class TestTextChunker:
    """Test cases for TextChunker"""
    
    @pytest.fixture
    def text_chunker(self):
        """Create TextChunker instance for testing"""
        config = {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "separators": ["\n\n", "\n", ".", "!", "?", ";", " "],
            "metadata_fields": ["document_id", "chunk_index", "source"]
        }
        return TextChunker(config)
    
    def test_initialization(self, text_chunker):
        """Test TextChunker initialization"""
        assert text_chunker is not None
        assert text_chunker.chunk_size == 100
        assert text_chunker.chunk_overlap == 20
    
    def test_chunk_document(self, text_chunker):
        """Test document chunking"""
        text = "This is a test document. It has multiple sentences. Each sentence should be processed separately. The chunking should work correctly."
        document_id = "test_doc"
        
        chunks = text_chunker.chunk_document(text, document_id)
        
        assert len(chunks) > 0
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("document_id" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all(chunk["document_id"] == document_id for chunk in chunks)
    
    def test_chunk_empty_text(self, text_chunker):
        """Test chunking empty text"""
        chunks = text_chunker.chunk_document("", "empty_doc")
        assert len(chunks) == 0
    
    def test_chunk_short_text(self, text_chunker):
        """Test chunking text shorter than chunk size"""
        short_text = "Short text."
        chunks = text_chunker.chunk_document(short_text, "short_doc")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
    
    def test_count_tokens(self, text_chunker):
        """Test token counting"""
        text = "This is a test sentence with multiple words."
        token_count = text_chunker.count_tokens(text)
        
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_get_chunk_statistics(self, text_chunker):
        """Test chunk statistics calculation"""
        chunks = [
            {"text_length": 100, "token_count": 20},
            {"text_length": 150, "token_count": 30},
            {"text_length": 200, "token_count": 40}
        ]
        
        stats = text_chunker.get_chunk_statistics(chunks)
        
        assert stats["total_chunks"] == 3
        assert stats["total_text_length"] == 450
        assert stats["total_tokens"] == 90
        assert stats["avg_chunk_size"] == 150.0
        assert stats["avg_token_count"] == 30.0
        assert stats["min_chunk_size"] == 100
        assert stats["max_chunk_size"] == 200
