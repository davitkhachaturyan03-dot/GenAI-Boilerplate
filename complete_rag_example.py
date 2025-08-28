#!/usr/bin/env python3
"""
Complete RAG/MCP/Graph RAG Example
==================================

This script demonstrates the complete flow of the RAG platform:
1. Document upload and processing
2. Vector similarity search
3. Graph RAG with entity extraction
4. Hybrid search (vector + text)
5. RAG query processing with MCP
6. Conversation management

Prerequisites:
- MCP server running (python start_mcp_server.py)
- Django server running (python manage.py runserver)
- API keys configured in .env file
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, List
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000/api"
MCP_SERVER_URL = "http://localhost:8080"

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "content": """
Machine learning is a subset of artificial intelligence that focuses on algorithms 
that can learn and make decisions from data. There are three main types of machine 
learning: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning uses labeled training data to learn a mapping from inputs to outputs. 
Common algorithms include linear regression, decision trees, and neural networks.

Unsupervised learning finds patterns in data without labeled examples. Clustering 
and dimensionality reduction are common unsupervised learning tasks.

Reinforcement learning involves an agent learning to make decisions by interacting 
with an environment and receiving rewards or penalties for its actions.
        """,
        "document_type": "educational",
        "metadata": {"subject": "AI", "difficulty": "beginner"}
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": """
Deep learning is a subset of machine learning that uses neural networks with 
multiple layers. These deep neural networks can learn complex patterns and 
representations from large amounts of data.

Key concepts in deep learning include:
- Neural networks with multiple hidden layers
- Backpropagation for training
- Activation functions like ReLU and sigmoid
- Convolutional Neural Networks (CNNs) for image processing
- Recurrent Neural Networks (RNNs) for sequence data
- Transformers for natural language processing

Popular deep learning frameworks include TensorFlow, PyTorch, and Keras. 
These frameworks make it easier to build and train deep learning models.
        """,
        "document_type": "educational",
        "metadata": {"subject": "AI", "difficulty": "intermediate"}
    },
    {
        "title": "Natural Language Processing with Transformers",
        "content": """
Natural Language Processing (NLP) has been revolutionized by transformer models. 
The transformer architecture, introduced in the "Attention is All You Need" paper, 
uses self-attention mechanisms to process text.

Key transformer models include:
- BERT (Bidirectional Encoder Representations from Transformers)
- GPT (Generative Pre-trained Transformer)  
- T5 (Text-to-Text Transfer Transformer)
- RoBERTa (Robustly Optimized BERT Pretraining Approach)

These models have achieved state-of-the-art results on many NLP tasks including:
- Text classification and sentiment analysis
- Question answering
- Language translation
- Text summarization
- Named entity recognition

The attention mechanism allows transformers to capture long-range dependencies 
in text, making them particularly effective for understanding context.
        """,
        "document_type": "educational",
        "metadata": {"subject": "NLP", "difficulty": "advanced"}
    }
]

SAMPLE_QUERIES = [
    "What are the main types of machine learning?",
    "How do transformers work in natural language processing?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the attention mechanism in deep learning",
    "What are popular deep learning frameworks?"
]


class RAGExampleRunner:
    """Complete RAG example runner with all platform features."""
    
    def __init__(self, api_base_url: str = API_BASE_URL, mcp_url: str = MCP_SERVER_URL):
        self.api_base = api_base_url
        self.mcp_url = mcp_url
        self.session = None
        self.uploaded_documents = []
        
    def log(self, message: str, level: str = "INFO"):
        """Pretty logging with timestamps."""
        timestamp = time.strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[94m",  # Blue
            "SUCCESS": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "RESET": "\033[0m"  # Reset
        }
        color = colors.get(level, colors["INFO"])
        print(f"{color}[{timestamp}] {level}: {message}{colors['RESET']}")
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.api_base}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.log(f"Request failed: {e}", "ERROR")
            return {"error": str(e)}
    
    def check_services(self) -> bool:
        """Check if all required services are running."""
        self.log("🔍 Checking service availability...")
        
        # Check Django API
        try:
            health = self.make_request("GET", "/rag/health/live/")
            if health.get("status") == "alive":
                self.log("✅ Django API is running", "SUCCESS")
            else:
                self.log("❌ Django API health check failed", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Django API is not accessible: {e}", "ERROR")
            return False
        
        # Check MCP Server
        try:
            mcp_response = requests.get(f"{self.mcp_url}/health", timeout=5)
            if mcp_response.status_code == 200:
                mcp_data = mcp_response.json()
                self.log("✅ MCP Server is running", "SUCCESS")
                self.log(f"   Available providers: {mcp_data.get('providers', {})}")
            else:
                self.log("❌ MCP Server is not responding", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ MCP Server is not accessible: {e}", "ERROR")
            return False
        
        return True
    
    def create_rag_session(self) -> bool:
        """Create a new RAG session."""
        self.log("🚀 Creating RAG session...")
        
        session_data = {
            "user_id": "demo_user",
            "system_prompt": "You are a helpful AI assistant specializing in machine learning and AI topics.",
            "temperature": 0.7,
            "max_context_length": 4000
        }
        
        result = self.make_request("POST", "/rag/sessions/", json=session_data)
        
        if "error" not in result:
            self.session = result
            self.log(f"✅ Session created: {result['session_id']}", "SUCCESS")
            return True
        else:
            self.log(f"❌ Failed to create session: {result['error']}", "ERROR")
            return False
    
    def upload_documents(self) -> bool:
        """Upload sample documents to the platform."""
        self.log("📄 Uploading sample documents...")
        
        for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
            self.log(f"   Uploading document {i}/{len(SAMPLE_DOCUMENTS)}: {doc['title']}")
            
            # Add chunking parameters
            doc_data = {
                **doc,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "extract_entities": True  # Enable entity extraction for Graph RAG
            }
            
            result = self.make_request("POST", "/vectors/upload/", json=doc_data)
            
            if "error" not in result:
                self.uploaded_documents.append(result)
                self.log(f"     ✅ Uploaded: {result['chunks_created']} chunks created")
            else:
                self.log(f"     ❌ Upload failed: {result['error']}", "ERROR")
                return False
        
        self.log("✅ All documents uploaded successfully", "SUCCESS")
        return True
    
    def demonstrate_vector_search(self):
        """Demonstrate vector similarity search."""
        self.log("🔍 Demonstrating Vector Search...")
        
        test_query = "What are neural networks and how do they work?"
        
        search_data = {
            "query_text": test_query,
            "top_k": 5,
            "similarity_threshold": 0.3,
            "distance_metric": "cosine"
        }
        
        result = self.make_request("POST", "/vectors/search/", json=search_data)
        
        if "error" not in result:
            self.log(f"✅ Found {result['results_count']} relevant chunks")
            
            for i, chunk in enumerate(result['results'][:3], 1):
                self.log(f"   Result {i}: {chunk['document_title']}")
                self.log(f"      Similarity: {chunk['similarity']:.3f}")
                self.log(f"      Content: {chunk['content'][:100]}...")
        else:
            self.log(f"❌ Vector search failed: {result['error']}", "ERROR")
    
    def demonstrate_hybrid_search(self):
        """Demonstrate hybrid search (vector + full-text)."""
        self.log("🔀 Demonstrating Hybrid Search...")
        
        test_query = "deep learning frameworks TensorFlow PyTorch"
        
        search_data = {
            "query_text": test_query,
            "top_k": 3,
            "text_weight": 0.4,
            "vector_weight": 0.6
        }
        
        result = self.make_request("POST", "/graph/hybrid/", json=search_data)
        
        if "error" not in result:
            self.log(f"✅ Hybrid search returned {result['results_count']} results")
            
            for i, chunk in enumerate(result['results'][:2], 1):
                self.log(f"   Result {i}: {chunk['document_title']}")
                self.log(f"      Vector Score: {chunk['vector_score']:.3f}")
                self.log(f"      Text Score: {chunk['text_score']:.3f}")
                self.log(f"      Hybrid Score: {chunk['hybrid_score']:.3f}")
        else:
            self.log(f"❌ Hybrid search failed: {result['error']}", "ERROR")
    
    def demonstrate_rag_queries(self):
        """Demonstrate different types of RAG queries."""
        self.log("💬 Demonstrating RAG Queries...")
        
        if not self.session:
            self.log("❌ No session available for RAG queries", "ERROR")
            return
        
        query_types = ["standard", "hybrid"]
        
        for query_type in query_types:
            self.log(f"   Testing {query_type.upper()} RAG...")
            
            test_query = SAMPLE_QUERIES[0]  # Use first sample query
            
            rag_data = {
                "query_text": test_query,
                "query_type": query_type,
                "session_id": self.session["session_id"],
                "top_k": 5,
                "temperature": 0.7,
                "max_response_tokens": 300
            }
            
            start_time = time.time()
            result = self.make_request("POST", "/rag/query/", json=rag_data)
            processing_time = time.time() - start_time
            
            if "error" not in result:
                self.log(f"     ✅ Query processed in {processing_time:.2f}s")
                self.log(f"     Response: {result['response_text'][:200]}...")
                self.log(f"     Context chunks: {result['context_chunks_count']}")
            else:
                self.log(f"     ❌ RAG query failed: {result['error']}", "ERROR")
    
    def demonstrate_conversation_flow(self):
        """Demonstrate multi-turn conversation."""
        self.log("💭 Demonstrating Conversation Flow...")
        
        if not self.session:
            self.log("❌ No session available for conversation", "ERROR")
            return
        
        conversation = [
            "What is machine learning?",
            "What are the main types you mentioned?",
            "Can you give me an example of supervised learning?",
            "How is this different from deep learning?"
        ]
        
        for i, query in enumerate(conversation, 1):
            self.log(f"   Turn {i}: {query}")
            
            rag_data = {
                "query_text": query,
                "query_type": "conversational",
                "session_id": self.session["session_id"],
                "top_k": 3,
                "temperature": 0.7
            }
            
            result = self.make_request("POST", "/rag/query/", json=rag_data)
            
            if "error" not in result:
                response = result['response_text']
                self.log(f"     Response: {response[:150]}...")
            else:
                self.log(f"     ❌ Conversation turn failed: {result['error']}", "ERROR")
            
            time.sleep(1)  # Brief pause between turns
    
    def demonstrate_mcp_tools(self):
        """Demonstrate MCP tool execution."""
        self.log("🔧 Demonstrating MCP Tools...")
        
        try:
            # List available tools
            tools_response = requests.get(f"{self.mcp_url}/v1/tools")
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                tools = tools_data.get("tools", [])
                self.log(f"✅ Found {len(tools)} available tools")
                
                for tool in tools:
                    self.log(f"   - {tool['name']}: {tool['description']}")
                
                # Test calculator tool
                if any(tool['name'] == 'calculator' for tool in tools):
                    self.log("   Testing calculator tool...")
                    
                    calc_data = {
                        "tool": "calculator",
                        "parameters": {
                            "expression": "2 + 2 * 3"
                        }
                    }
                    
                    tool_response = requests.post(
                        f"{self.mcp_url}/v1/tools/execute",
                        json=calc_data
                    )
                    
                    if tool_response.status_code == 200:
                        tool_result = tool_response.json()
                        self.log(f"     Calculator result: {tool_result.get('result')}")
                    
            else:
                self.log("❌ Could not retrieve MCP tools", "ERROR")
        
        except Exception as e:
            self.log(f"❌ MCP tools demonstration failed: {e}", "ERROR")
    
    def test_health_endpoints(self):
        """Test all health check endpoints."""
        self.log("🏥 Testing Health Endpoints...")
        
        health_endpoints = [
            ("/rag/health/live/", "Liveness Check"),
            ("/rag/health/ready/", "Readiness Check"),
            ("/rag/health/", "Full Health Check"),
            ("/rag/ping/", "Simple Ping")
        ]
        
        for endpoint, name in health_endpoints:
            result = self.make_request("GET", endpoint)
            if "error" not in result:
                status = result.get("status", "unknown")
                self.log(f"   ✅ {name}: {status}")
            else:
                self.log(f"   ❌ {name}: Failed", "ERROR")
    
    def cleanup(self):
        """Clean up resources."""
        self.log("🧹 Cleaning up...")
        # In a real scenario, you might want to delete test documents
        # or close sessions, but for this demo we'll just log
        self.log("✅ Cleanup completed")
    
    def run_complete_example(self):
        """Run the complete RAG platform example."""
        self.log("🎯 Starting Complete RAG Platform Example", "SUCCESS")
        self.log("=" * 60)
        
        try:
            # Step 1: Check services
            if not self.check_services():
                self.log("❌ Service check failed. Please ensure all services are running.", "ERROR")
                return False
            
            # Step 2: Create RAG session
            if not self.create_rag_session():
                return False
            
            # Step 3: Upload documents
            if not self.upload_documents():
                return False
            
            # Give some time for document processing
            self.log("⏳ Waiting for document processing...")
            time.sleep(3)
            
            # Step 4: Demonstrate vector search
            self.demonstrate_vector_search()
            
            # Step 5: Demonstrate hybrid search
            self.demonstrate_hybrid_search()
            
            # Step 6: Demonstrate RAG queries
            self.demonstrate_rag_queries()
            
            # Step 7: Demonstrate conversation flow
            self.demonstrate_conversation_flow()
            
            # Step 8: Demonstrate MCP tools
            self.demonstrate_mcp_tools()
            
            # Step 9: Test health endpoints
            self.test_health_endpoints()
            
            # Step 10: Cleanup
            self.cleanup()
            
            self.log("🎉 Complete RAG example finished successfully!", "SUCCESS")
            return True
            
        except KeyboardInterrupt:
            self.log("❌ Example interrupted by user", "WARNING")
            return False
        except Exception as e:
            self.log(f"❌ Example failed with error: {e}", "ERROR")
            return False


def print_setup_instructions():
    """Print setup instructions for running the example."""
    print("\n" + "=" * 80)
    print("🚀 RAG Platform Complete Example")
    print("=" * 80)
    print("\n📋 Prerequisites:")
    print("1. Set up your API keys in the .env file:")
    print("   - OPENAI_API_KEY=your_openai_key")
    print("   - ANTHROPIC_API_KEY=your_anthropic_key")
    print("   - (or other provider keys)")
    print("\n2. Start the required services:")
    print("   Terminal 1: python start_mcp_server.py")
    print("   Terminal 2: python manage.py runserver")
    print("\n3. Run this example:")
    print("   python complete_rag_example.py")
    print("\n🔗 Services should be running on:")
    print("   - MCP Server: http://localhost:8080")
    print("   - Django API: http://localhost:8000")
    print("\n" + "=" * 80 + "\n")


def main():
    """Main function to run the complete example."""
    print_setup_instructions()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Usage: python complete_rag_example.py [--skip-check]")
        print("Options:")
        print("  --skip-check    Skip service availability check")
        return
    
    skip_check = "--skip-check" in sys.argv
    
    try:
        runner = RAGExampleRunner()
        
        if not skip_check:
            input("Press Enter when all services are running (or Ctrl+C to exit)...")
        
        success = runner.run_complete_example()
        
        if success:
            print("\n🎉 Example completed successfully!")
            print("🔍 Check the logs above for detailed results.")
            print("📊 You can also check the Django admin or API endpoints directly.")
        else:
            print("\n❌ Example failed. Check the error messages above.")
            print("💡 Make sure all services are running and API keys are configured.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()