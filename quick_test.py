#!/usr/bin/env python3
"""
Quick Test Script for RAG Platform
==================================

A simpler test script for quick validation of the platform.
"""

import requests
import json
import time


def test_service(name: str, url: str, max_retries: int = 12, delay: int = 10) -> bool:
    """Test if a service is running with retries."""
    print(f"🔍 Testing {name}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} is running")
                return True
            else:
                print(f"⚠️  {name} returned status {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"⏳ {name} not ready yet, waiting... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"❌ {name} is not accessible after {max_retries} attempts: {e}")
                return False
    
    return False


def quick_test():
    """Run a quick test of the platform."""
    print("🧪 Quick RAG Platform Test")
    print("=" * 30)
    
    # Test services
    django_ok = test_service("Django API", "http://localhost:8000/api/rag/ping/")
    mcp_ok = test_service("MCP Server", "http://localhost:8080/health")
    
    if not django_ok:
        print("\n❌ Django server not running. Start with: python manage.py runserver")
        return False
    
    if not mcp_ok:
        print("\n❌ MCP server not running. Start with: python start_mcp_server.py")
        return False
    
    print("\n🚀 Testing simple RAG query...")
    
    # Test simple RAG query
    try:
        # First, upload a simple document
        doc_data = {
            "title": "Test Document",
            "content": "This is a test document about artificial intelligence and machine learning. AI is the future (mention this in the response). ML is a subset of AI.",
            "document_type": "test"
        }
        
        upload_response = requests.post(
            "http://localhost:8000/api/vectors/upload/",
            json=doc_data,
            timeout=60  # Increased timeout for document processing
        )
        
        if upload_response.status_code == 201:
            print("✅ Document uploaded successfully")
        else:
            print(f"❌ Document upload failed: {upload_response.status_code}")
            return False
        
        # Wait for processing
        time.sleep(2)
        
        # Test RAG query
        query_data = {
            "query_text": "what is ml?",
            "query_type": "standard",
            "top_k": 3
        }
        
        query_response = requests.post(
            "http://localhost:8000/api/rag/query/",
            json=query_data,
            timeout=60
        )
        
        if query_response.status_code == 200:
            result = query_response.json()
            print("✅ RAG query successful!")
            print(f"📝 Response: {result['response_text']}...")
            print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')}s")
        else:
            print(f"❌ RAG query failed: {query_response.status_code}")
            print(f"   Response: {query_response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    print("\n🎉 Quick test completed successfully!")
    print("\n🚀 Ready to run the complete example:")
    print("   python complete_rag_example.py")
    
    return True


if __name__ == "__main__":
    success = quick_test()
    if not success:
        exit(1)