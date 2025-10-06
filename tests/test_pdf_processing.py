"""
Tests for PDF processing module
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from modules.pdf_processing import PDFProcessor


class TestPDFProcessor:
    """Test cases for PDFProcessor"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pdf_processor(self, temp_dir):
        """Create PDFProcessor instance for testing"""
        config = {
            "input_dir": temp_dir,
            "output_dir": f"{temp_dir}/processed",
            "supported_formats": [".pdf"],
            "text_cleaning": {
                "remove_headers_footers": True,
                "remove_figure_captions": True,
                "normalize_whitespace": True,
                "min_text_length": 10
            }
        }
        return PDFProcessor(config)
    
    def test_initialization(self, pdf_processor):
        """Test PDFProcessor initialization"""
        assert pdf_processor is not None
        assert pdf_processor.input_dir is not None
        assert pdf_processor.output_dir is not None
    
    def test_clean_text(self, pdf_processor):
        """Test text cleaning functionality"""
        # Test text with excessive whitespace
        dirty_text = "This   is   a   test   with   lots   of   spaces.\n\n\n\nMultiple\n\n\n\nbreaks."
        cleaned = pdf_processor.clean_text(dirty_text)
        
        assert "   " not in cleaned  # No triple spaces
        assert "\n\n\n" not in cleaned  # No triple newlines
        assert len(cleaned) < len(dirty_text)  # Should be shorter
    
    def test_save_cleaned_text(self, pdf_processor, temp_dir):
        """Test saving cleaned text"""
        doc_id = "test_doc"
        text = "This is a test document with some content."
        metadata = {"source": "test"}
        
        result = pdf_processor.save_cleaned_text(doc_id, text, metadata)
        
        assert result != ""  # Should return file path
        assert Path(result).exists()  # File should exist
        
        # Check file content
        with open(result, 'r', encoding='utf-8') as f:
            import json
            data = json.load(f)
            assert data["doc_id"] == doc_id
            assert data["text"] == text
            assert data["metadata"] == metadata
    
    def test_process_single_pdf_nonexistent(self, pdf_processor):
        """Test processing non-existent PDF file"""
        result = pdf_processor.process_single_pdf("nonexistent.pdf")
        
        assert result["success"] == False
        assert "File not found" in result["error"]
    
    def test_get_processing_stats_empty(self, pdf_processor):
        """Test getting stats from empty processor"""
        stats = pdf_processor.get_processing_stats()
        
        assert stats["total_processed"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["total_text_length"] == 0
        assert stats["average_text_length"] == 0
