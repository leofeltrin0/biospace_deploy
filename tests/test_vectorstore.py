"""
Tests for vectorstore module
"""

import pytest
import tempfile
import shutil
import numpy as np
from modules.vectorstore import VectorStore


class TestVectorStore:
    """Test cases for VectorStore"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vectorstore(self, temp_dir):
        """Create VectorStore instance for testing"""
        return VectorStore(backend="faiss", persist_dir=temp_dir)
    
    def test_initialization(self, vectorstore):
        """Test VectorStore initialization"""
        assert vectorstore is not None
        assert vectorstore.backend == "faiss"
        assert vectorstore.persist_dir is not None
    
    def test_add_embeddings(self, vectorstore):
        """Test adding embeddings to vectorstore"""
        embeddings = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "text": "Test text 1",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {"source": "test"}
            },
            {
                "chunk_id": "chunk_2", 
                "document_id": "doc_1",
                "text": "Test text 2",
                "embedding": [0.6, 0.7, 0.8, 0.9, 1.0],
                "metadata": {"source": "test"}
            }
        ]
        
        result = vectorstore.add_embeddings(embeddings)
        assert result == True
    
    def test_query_empty_store(self, vectorstore):
        """Test querying empty vectorstore"""
        results = vectorstore.query("test query", top_k=5)
        assert len(results) == 0
    
    def test_save_and_load(self, vectorstore):
        """Test saving and loading vectorstore"""
        # Add some embeddings
        embeddings = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1", 
                "text": "Test text",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {"source": "test"}
            }
        ]
        
        vectorstore.add_embeddings(embeddings)
        
        # Save
        save_result = vectorstore.save()
        assert save_result == True
        
        # Load
        load_result = vectorstore.load()
        assert load_result == True
    
    def test_get_stats(self, vectorstore):
        """Test getting vectorstore statistics"""
        stats = vectorstore.get_stats()
        
        assert "backend" in stats
        assert stats["backend"] == "faiss"
        assert "total_vectors" in stats
        assert "dimension" in stats
    
    def test_clear(self, vectorstore):
        """Test clearing vectorstore"""
        # Add some data first
        embeddings = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "text": "Test text",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {"source": "test"}
            }
        ]
        
        vectorstore.add_embeddings(embeddings)
        
        # Clear
        clear_result = vectorstore.clear()
        assert clear_result == True
        
        # Check stats after clearing
        stats = vectorstore.get_stats()
        assert stats["total_vectors"] == 0
