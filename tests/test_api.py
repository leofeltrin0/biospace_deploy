"""
Tests for FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from modules.api.app import app


class TestAPI:
    """Test cases for FastAPI application"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_query_endpoint_missing_data(self, client):
        """Test query endpoint with missing data"""
        response = client.get("/query", params={"query": "test query"})
        # Should return 500 since components aren't initialized in test
        assert response.status_code == 500
    
    def test_graph_endpoint_missing_data(self, client):
        """Test graph endpoint with missing data"""
        response = client.get("/graph")
        # Should return 500 since components aren't initialized in test
        assert response.status_code == 500
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "api" in data
        assert "timestamp" in data
    
    def test_profiles_endpoint_missing_data(self, client):
        """Test profiles endpoint with missing data"""
        response = client.get("/profiles")
        # Should return 500 since components aren't initialized in test
        assert response.status_code == 500
    
    def test_404_handler(self, client):
        """Test 404 error handler"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "Endpoint not found" in data["error"]
