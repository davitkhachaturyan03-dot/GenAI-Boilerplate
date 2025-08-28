-- PostgreSQL setup script for RAG platform
-- This runs automatically when the container starts

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user (optional - if you want a specific user for the app)
-- CREATE USER rag_user WITH PASSWORD 'your_password';
-- GRANT ALL PRIVILEGES ON DATABASE rag_database TO rag_user;
-- GRANT ALL ON SCHEMA public TO rag_user;

-- Verify pgvector installation
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Create indexes for better performance (run after migrations)
-- These will be created automatically by Django, but you can optimize them

-- Example index creation (uncomment after running migrations):

/*
-- Vector similarity indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_cosine 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_l2 
ON document_chunks USING ivfflat (embedding vector_l2_ops) 
WITH (lists = 100);

-- For HNSW index (better for high-dimensional vectors):
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Entity embedding indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_embedding_cosine
ON entities USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Search query embedding indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_search_queries_embedding_cosine
ON search_queries USING ivfflat (query_embedding vector_cosine_ops) 
WITH (lists = 100);

-- Community summary embedding indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_communities_summary_embedding_cosine
ON communities USING ivfflat (summary_embedding vector_cosine_ops) 
WHERE summary_embedding IS NOT NULL
WITH (lists = 50);

-- Full-text search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_fts
ON document_chunks USING gin(to_tsvector('english', content));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_content_fts
ON documents USING gin(to_tsvector('english', content));

-- Performance optimization indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_type_active
ON documents (document_type, is_active, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_type_confidence
ON entities (entity_type, confidence_score, is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_relationships_type_weight
ON relationships (relationship_type, weight, is_active);

-- Partial indexes for active records only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_active_embedding
ON document_chunks (embedding) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entities_active_embedding
ON entities (embedding) WHERE is_active = true;
*/

-- Show current database configuration
SELECT 
    name,
    setting,
    unit,
    category,
    short_desc
FROM pg_settings 
WHERE name IN (
    'shared_preload_libraries',
    'max_connections',
    'shared_buffers',
    'effective_cache_size',
    'work_mem',
    'maintenance_work_mem'
);

-- Performance recommendations for vector operations:
-- Add to postgresql.conf:
-- shared_preload_libraries = 'vector'
-- shared_buffers = 256MB (or 25% of RAM)
-- effective_cache_size = 1GB (or 75% of RAM)
-- work_mem = 64MB
-- maintenance_work_mem = 256MB

-- Database setup complete!
-- Next steps:
-- 1. Update your .env file with database credentials
-- 2. Run: python create_migrations.py
-- 3. Run: python manage.py migrate
-- 4. Create indexes after migrations using the commented SQL above