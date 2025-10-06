#!/usr/bin/env python3
"""
Simple health check test for the API
"""
import requests
import time
import subprocess
import sys
import os

def test_health_endpoint():
    """Test the /health endpoint"""
    print("Testing /health endpoint...")
    
    # Try to start the server in background
    try:
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "modules.api.app:app", 
            "--host", "127.0.0.1", 
            "--port", "8001"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8001/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False
    finally:
        # Clean up
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

if __name__ == "__main__":
    success = test_health_endpoint()
    sys.exit(0 if success else 1)
