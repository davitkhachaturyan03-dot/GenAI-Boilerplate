# RAG/MCP and Graph RAG Integration Platform

A scalable, production-ready platform for Retrieval-Augmented Generation (RAG) with Model Context Protocol (MCP) and Graph RAG capabilities, built with Django and PostgreSQL vector storage.

## Features

### Core RAG Capabilities
- **Vector-based retrieval** using pgvector for similarity search
- **Multiple embedding models** support (Sentence Transformers, OpenAI, etc.)
- **Hybrid search** combining vector similarity and full-text search
- **Conversation context** management for multi-turn interactions
- **Flexible chunking** strategies with configurable overlap

### Graph RAG
- **Knowledge graph construction** with entities and relationships
- **Community detection** for hierarchical graph organization
- **Graph-aware retrieval** using entity and relationship context
- **Temporal relationships** for time-aware knowledge graphs
- **Multiple graph algorithms** (PageRank, centrality measures)

### MCP Integration
- **Model Context Protocol** client for LLM integration
- **Multi-provider support** (Claude, OpenAI, Custom APIs)
- **Async request handling** with streaming support
- **Tool execution** framework for function calling
- **Request/response tracking** and analytics

### Scalability Features
- **PostgreSQL** with pgvector extension for vector operations
- **Generic foreign keys** for flexible data relationships
- **Batch processing** capabilities
- **Configurable indexing** (IVFFlat, HNSW)
- **RESTful APIs** with comprehensive serialization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Core      │    │   Graph RAG     │    │ MCP Integration │
│                 │    │                 │    │                 │
│ • RAG Engine    │    │ • Knowledge     │    │ • MCP Client    │
│ • Sessions      │    │   Graphs        │    │ • Tool Executor │
│ • Queries       │    │ • Entities      │    │ • Request Mgmt  │
│ • Context Mgmt  │    │ • Relationships │    │ • Multi-provider│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Vector Store                       │
         │                                                 │
         │ • Documents & Chunks    • Vector Search         │
         │ • pgvector Integration  • Similarity Matching   │
         │ • Embedding Storage     • Hybrid Retrieval      │
         │ • Index Management      • Batch Operations      │
         └─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- Redis (optional, for Celery)

### Installation

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd rag-graph-mcp-platform
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

2. **Database setup**
```bash
# Install pgvector extension in PostgreSQL
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Copy environment configuration
cp .env.example .env
# Edit .env with your database credentials and API keys
```

3. **Run migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

4. **Create superuser**
```bash
python manage.py createsuperuser
```

5. **Start the server**
```bash
python manage.py runserver
```

## API Endpoints

### RAG Operations
- `POST /api/rag/query/` - Process RAG queries
- `GET /api/rag/sessions/` - List RAG sessions
- `POST /api/rag/sessions/` - Create new session
- `POST /api/rag/feedback/` - Submit feedback

### Vector Search
- `POST /api/vectors/search/` - Vector similarity search
- `POST /api/vectors/upload/` - Upload and process documents

### Graph RAG
- `POST /api/graph/query/` - Graph RAG queries
- `POST /api/graph/hybrid/` - Hybrid graph+vector search

### MCP Integration
- `GET /api/mcp/servers/` - List MCP servers
- `POST /api/mcp/request/` - Send MCP requests
- `GET /api/mcp/tools/` - List available tools

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Core Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
CHUNK_SIZE=512

# Database
DB_NAME=rag_database
DB_USER=postgres
DB_PASSWORD=your_password

# MCP Server
MCP_SERVER_URL=http://localhost:8080
MCP_API_KEY=your_api_key
```

### Database Schema

The platform uses a scalable database design:

- **Documents & Chunks**: Hierarchical document storage with vector embeddings
- **Knowledge Graphs**: Entity-relationship models with temporal support
- **RAG Sessions**: Conversation context and user session management
- **MCP Integration**: Request tracking and server configuration

## Usage Examples

### Basic RAG Query
```python
import requests

response = requests.post('http://localhost:8000/api/rag/query/', json={
    "query_text": "What is the capital of France?",
    "query_type": "standard",
    "top_k": 5,
    "temperature": 0.7
})
print(response.json()['response_text'])
```

### Graph RAG Query
```python
response = requests.post('http://localhost:8000/api/graph/query/', json={
    "query_text": "Tell me about relationships between entities",
    "knowledge_graph_id": 1,
    "max_entities": 10,
    "include_relationships": True
})
```

### Document Upload
```python
response = requests.post('http://localhost:8000/api/vectors/upload/', json={
    "title": "My Document",
    "content": "Document content here...",
    "document_type": "article",
    "chunk_size": 512,
    "extract_entities": True
})
```

## Advanced Features

### Custom Embedding Models
```python
# In settings.py
EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
VECTOR_DIMENSION = 768
```

### Graph RAG Configuration
```python
# Custom graph algorithms
graph_engine.compute_node_importance(nodes, algorithm='pagerank')

# Community detection
communities = Community.objects.filter(level=0)  # Top-level communities
```

### MCP Server Setup
```python
# Configure multiple MCP servers
servers = [
    {
        "name": "Claude",
        "url": "https://api.anthropic.com",
        "server_type": "claude"
    },
    {
        "name": "OpenAI",
        "url": "https://api.openai.com",
        "server_type": "openai"
    }
]
```

## Production Deployment

### Docker Configuration
```bash
# Build and run with Docker
docker-compose up --build
```

### Performance Optimization
- Use pgvector indexes for large-scale retrieval
- Configure Redis for session management
- Implement async processing with Celery
- Enable database query optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the API examples