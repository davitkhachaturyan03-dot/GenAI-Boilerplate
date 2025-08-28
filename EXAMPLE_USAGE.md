# Complete RAG Platform Example Usage Guide

This guide demonstrates how to run a complete example covering the entire RAG/MCP/Graph RAG flow.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the complete example with automatic setup
./run_example.sh
```

### Option 2: Manual Setup
```bash
# Terminal 1: Start MCP server
python start_mcp_server.py

# Terminal 2: Start Django server
python manage.py runserver

# Terminal 3: Run the example
python complete_rag_example.py
```

### Option 3: Quick Test Only
```bash
# Just test basic connectivity
python quick_test.py
# OR
./run_example.sh --quick-test
```

## 📋 Prerequisites

### 1. Environment Configuration
Make sure your `.env` file has at least one API key configured:

```bash
# Required: At least one LLM provider API key
OPENAI_API_KEY=sk-your-openai-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
# OR  
COHERE_API_KEY=your-cohere-key-here

# Database (if using PostgreSQL)
DB_NAME=rag_database
DB_USER=postgres
DB_PASSWORD=password

# Other configurations are optional
MCP_SERVER_URL=http://localhost:8080
DEFAULT_LLM_MODEL=gpt-3.5-turbo
```

### 2. Python Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-mcp.txt
```

### 3. Database Setup (if using PostgreSQL)
```bash
# Run migrations
python manage.py migrate
```

## 🎯 What the Complete Example Does

### 1. **Service Health Checks** ✅
- Verifies MCP server is running and responsive
- Checks Django API endpoints are accessible
- Validates LLM provider connectivity

### 2. **Document Upload & Processing** 📄
- Uploads 3 sample educational documents about AI/ML
- Chunks documents with configurable size and overlap
- Generates embeddings for vector search
- Extracts entities for Graph RAG (if enabled)

### 3. **Vector Similarity Search** 🔍
- Demonstrates semantic search using embeddings
- Shows similarity scores and ranking
- Tests different distance metrics (cosine, L2, etc.)

### 4. **Hybrid Search** 🔀
- Combines vector similarity with full-text search
- Configurable weights for different search methods
- Demonstrates enhanced retrieval accuracy

### 5. **RAG Query Processing** 💬
- Standard RAG: Vector retrieval + LLM generation
- Conversational RAG: Multi-turn context awareness
- Hybrid RAG: Combined vector + graph retrieval

### 6. **Graph RAG Features** 🕸️
- Entity extraction and relationship mapping
- Community detection in knowledge graphs
- Graph-aware context retrieval
- Path finding between entities

### 7. **MCP Tool Execution** 🔧
- Lists available MCP tools
- Demonstrates tool execution (calculator example)
- Shows function calling capabilities

### 8. **Conversation Management** 💭
- Multi-turn conversations with context
- Session-based query tracking
- Response history and context building

## 📊 Sample Output

```
🎯 Starting Complete RAG Platform Example
========================================

[14:32:15] INFO: 🔍 Checking service availability...
[14:32:15] SUCCESS: ✅ Django API is running
[14:32:15] SUCCESS: ✅ MCP Server is running
           Available providers: {'openai': True, 'anthropic': False}

[14:32:16] INFO: 🚀 Creating RAG session...
[14:32:16] SUCCESS: ✅ Session created: a1b2c3d4-e5f6-7890-abcd-ef1234567890

[14:32:16] INFO: 📄 Uploading sample documents...
[14:32:16] INFO:    Uploading document 1/3: Introduction to Machine Learning
[14:32:17] INFO:      ✅ Uploaded: 4 chunks created
[14:32:17] INFO:    Uploading document 2/3: Deep Learning Fundamentals
[14:32:18] INFO:      ✅ Uploaded: 3 chunks created
...

[14:32:25] INFO: 🔍 Demonstrating Vector Search...
[14:32:26] SUCCESS: ✅ Found 5 relevant chunks
[14:32:26] INFO:    Result 1: Deep Learning Fundamentals
[14:32:26] INFO:       Similarity: 0.856
[14:32:26] INFO:       Content: Deep learning is a subset of machine learning...

[14:32:26] INFO: 💬 Demonstrating RAG Queries...
[14:32:27] INFO:    Testing STANDARD RAG...
[14:32:29] INFO:      ✅ Query processed in 2.34s
[14:32:29] INFO:      Response: Machine learning has three main types: supervised learning, which uses labeled data...
[14:32:29] INFO:      Context chunks: 3

🎉 Complete RAG example finished successfully!
```

## 🛠️ Troubleshooting

### Common Issues

**1. "MCP Server not accessible"**
```bash
# Check if MCP server is running
curl http://localhost:8080/health

# Start MCP server manually
python start_mcp_server.py
```

**2. "No API keys configured"**
```bash
# Check your .env file
cat .env | grep API_KEY

# Add at least one API key
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

**3. "Django server not responding"**
```bash
# Check Django server
curl http://localhost:8000/api/rag/ping/

# Start Django server
python manage.py runserver
```

**4. "Database connection error"**
```bash
# Run migrations
python manage.py migrate

# Check database settings in .env
```

### Log Files
When using `./run_example.sh`, check these log files for debugging:

- `mcp_server.log` - MCP server output
- `django_server.log` - Django server output  
- `django_migrate.log` - Database migration output

### Manual Cleanup
```bash
# Stop background processes
./run_example.sh --cleanup

# Or manually kill processes
pkill -f "start_mcp_server"
pkill -f "manage.py runserver"
```

## 🔧 Customization

### Modify Sample Documents
Edit the `SAMPLE_DOCUMENTS` list in `complete_rag_example.py`:

```python
SAMPLE_DOCUMENTS = [
    {
        "title": "Your Document Title",
        "content": "Your document content here...",
        "document_type": "custom",
        "metadata": {"custom_field": "custom_value"}
    }
]
```

### Change Query Examples
Modify the `SAMPLE_QUERIES` list:

```python
SAMPLE_QUERIES = [
    "Your custom question here?",
    "Another question about your domain?",
]
```

### Configure LLM Parameters
In the RAG query examples:

```python
rag_data = {
    "query_text": query,
    "query_type": "standard",
    "temperature": 0.5,  # Lower for more focused responses
    "max_response_tokens": 500,  # Longer responses
    "top_k": 10,  # More context chunks
}
```

## 📈 Advanced Usage

### Run with Docker
```bash
# Start everything with Docker
make dev

# Run example against Docker services
python complete_rag_example.py
```

### Production Testing
```bash
# Use production compose
make prod

# Test against production setup
DJANGO_URL=http://your-domain.com python complete_rag_example.py
```

### Performance Testing
```bash
# Run multiple iterations
for i in {1..10}; do
    echo "Run $i"
    python complete_rag_example.py --skip-check
done
```

## 🎉 Next Steps

After running the complete example:

1. **Explore the API**: Check `http://localhost:8000/api/` for all endpoints
2. **Admin Interface**: Visit `http://localhost:8000/admin/` to see data models
3. **Custom Integration**: Use the platform APIs in your own applications
4. **Scale Up**: Deploy with Docker in production
5. **Extend**: Add custom tools, models, or document types

For more advanced usage, check the individual component documentation and API references.