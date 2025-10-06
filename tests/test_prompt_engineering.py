"""
Tests for prompt engineering module
"""

import pytest
import json
from modules.prompt_engineering import PromptEngineer


class TestPromptEngineer:
    """Test cases for PromptEngineer"""
    
    @pytest.fixture
    def prompt_engineer(self):
        """Create PromptEngineer instance for testing"""
        config = {
            "reference_extraction": True,
            "theme_classification": True,
            "structured_output": True,
            "themes": ["biotechnology", "neuroscience", "biochemistry", "ecology", "microbiology", "genetics"]
        }
        return PromptEngineer(config)
    
    def test_initialization(self, prompt_engineer):
        """Test PromptEngineer initialization"""
        assert prompt_engineer is not None
        assert prompt_engineer.reference_extraction == True
        assert prompt_engineer.theme_classification == True
        assert prompt_engineer.structured_output == True
        assert len(prompt_engineer.themes) == 6
    
    def test_build_prompt(self, prompt_engineer):
        """Test prompt building"""
        query = "What are the effects of microgravity on C. elegans?"
        chunks = [
            {
                "document_id": "test_doc_1",
                "chunk_id": "chunk_1",
                "text": "Microgravity affects C. elegans development and behavior.",
                "metadata": {"authors": "Smith et al.", "date": "2023"}
            }
        ]
        user_type = "scientist"
        
        prompt = prompt_engineer.build_prompt(query, chunks, user_type)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert "NASA expert assistant" in prompt
        assert "JSON format" in prompt
        assert "references" in prompt
        assert "theme" in prompt
    
    def test_parse_model_response_valid_json(self, prompt_engineer):
        """Test parsing valid JSON response"""
        valid_response = '''
        {
          "answer": "Microgravity affects C. elegans development and behavior.",
          "references": [
            {
              "file": "test_doc.pdf",
              "authors": "Smith et al.",
              "date": "2023",
              "relevance_score": 0.8
            }
          ],
          "theme": "biotechnology",
          "confidence": 0.9,
          "key_findings": ["Microgravity impacts development", "Behavioral changes observed"]
        }
        '''
        
        result = prompt_engineer.parse_model_response(valid_response)
        
        assert result["answer"] == "Microgravity affects C. elegans development and behavior."
        assert len(result["references"]) == 1
        assert result["theme"] == "biotechnology"
        assert result["confidence"] == 0.9
        assert len(result["key_findings"]) == 2
    
    def test_parse_model_response_invalid_json(self, prompt_engineer):
        """Test parsing invalid JSON response"""
        invalid_response = "This is not JSON at all"
        
        result = prompt_engineer.parse_model_response(invalid_response)
        
        assert "answer" in result
        assert "references" in result
        assert "theme" in result
        assert "confidence" in result
        assert "key_findings" in result
    
    def test_classify_theme_from_content(self, prompt_engineer):
        """Test theme classification from content"""
        # Test biotechnology content
        biotech_content = "Genetic engineering and recombinant DNA technology"
        theme = prompt_engineer._classify_theme_from_content(biotech_content)
        assert theme == "biotechnology"
        
        # Test neuroscience content
        neuro_content = "Neural networks and brain development"
        theme = prompt_engineer._classify_theme_from_content(neuro_content)
        assert theme == "neuroscience"
        
        # Test microbiology content
        micro_content = "Bacterial growth and microbial communities"
        theme = prompt_engineer._classify_theme_from_content(micro_content)
        assert theme == "microbiology"
    
    def test_extract_authors_from_metadata(self, prompt_engineer):
        """Test author extraction from metadata"""
        metadata = {"authors": "Smith, J. et al."}
        text = "Some research content"
        
        authors = prompt_engineer._extract_authors_from_metadata(metadata, text)
        assert authors == "Smith, J. et al."
    
    def test_extract_authors_from_text(self, prompt_engineer):
        """Test author extraction from text"""
        metadata = {}
        text = "Authors: Johnson, A. and Brown, B. (2023)"
        
        authors = prompt_engineer._extract_authors_from_metadata(metadata, text)
        assert "Johnson" in authors
    
    def test_extract_date_from_metadata(self, prompt_engineer):
        """Test date extraction from metadata"""
        metadata = {"date": "2023-01-15"}
        text = "Some research content"
        
        date = prompt_engineer._extract_date_from_metadata(metadata, text)
        assert date == "2023-01-15"
    
    def test_extract_date_from_text(self, prompt_engineer):
        """Test date extraction from text"""
        metadata = {}
        text = "Published in 2023, this study shows..."
        
        date = prompt_engineer._extract_date_from_metadata(metadata, text)
        assert date == "2023"
    
    def test_validate_response_structure(self, prompt_engineer):
        """Test response structure validation"""
        incomplete_response = {
            "answer": "Test answer",
            "theme": "invalid_theme"
        }
        
        validated = prompt_engineer._validate_response_structure(incomplete_response)
        
        assert validated["answer"] == "Test answer"
        assert validated["theme"] in prompt_engineer.themes  # Should be corrected
        assert "references" in validated
        assert "confidence" in validated
        assert "key_findings" in validated
    
    def test_get_theme_statistics(self, prompt_engineer):
        """Test theme statistics calculation"""
        responses = [
            {"theme": "biotechnology"},
            {"theme": "neuroscience"},
            {"theme": "biotechnology"},
            {"theme": "microbiology"}
        ]
        
        stats = prompt_engineer.get_theme_statistics(responses)
        
        assert stats["total_responses"] == 4
        assert stats["theme_counts"]["biotechnology"] == 2
        assert stats["theme_counts"]["neuroscience"] == 1
        assert stats["theme_counts"]["microbiology"] == 1
        assert stats["most_common_theme"] == "biotechnology"
    
    def test_adaptive_rag_pipeline_fallback(self, prompt_engineer):
        """Test adaptive RAG pipeline without OpenAI client"""
        query = "What is space biology?"
        chunks = [
            {
                "document_id": "test_doc",
                "chunk_id": "chunk_1",
                "text": "Space biology studies life in space environments.",
                "metadata": {}
            }
        ]
        user_type = "scientist"
        
        # Test without OpenAI client (fallback mode)
        prompt_engineer.openai_client = None
        
        result = prompt_engineer.adaptive_rag_pipeline(query, chunks, user_type)
        
        assert "answer" in result
        assert "references" in result
        assert "theme" in result
        assert "confidence" in result
        assert "key_findings" in result
        assert result["user_type"] == user_type
        assert result["chunks_used"] == 1
