import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
from pgvector.django import CosineDistance

from .models import RAGSession, RAGQuery, RAGContext
from .llm_client import llm_client
from vector_store.models import DocumentChunk, SearchQuery, SearchResult
from graph_rag.models import KnowledgeGraph
from graph_rag.graph_rag_engine import GraphRAGEngine


class RAGEngine:
    def __init__(self, session: RAGSession):
        self.session = session
        self.embedding_model = None
        self.graph_engine = None
        self._load_models()
    
    def _load_models(self):
        """Load graph engine if needed."""
        self.embedding_model = None  # Will use OpenAI embeddings via MCP
        
        if self.session.knowledge_graph:
            self.graph_engine = GraphRAGEngine(self.session.knowledge_graph)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for the given text using OpenAI via MCP."""
        print("Generating embedding for text... inside embed_text of RAGEngine")
        embedding_response = llm_client.sync_create_embedding([text])
        if 'error' in embedding_response:
            print(f"Error generating embedding: {embedding_response['error']['message']}")
            raise Exception(f"Embedding generation failed: {embedding_response['error']['message']}")
        return embedding_response['data'][0]['embedding']
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Retrieve relevant document chunks using vector similarity."""
        print()
        query_embedding = self.embed_text(query)
        print("Query embedding generated.")
        queryset = DocumentChunk.objects.filter(
            is_active=True,
            document__is_active=True
        ).annotate(
            similarity=1 - CosineDistance('embedding', query_embedding)
        ).filter(
            similarity__gte=similarity_threshold
        ).order_by('-similarity')

        print(f"Found {queryset.count()} chunks above similarity threshold {similarity_threshold}.")
        print(f"\n\nqueryset: \n{queryset}\n\n")
        # Apply additional filters if provided
        print(f"Applying filters: {filters}")
        if filters:
            if 'document_types' in filters and filters['document_types']:
                queryset = queryset.filter(
                    document__document_type__in=filters['document_types']
                )
                print(f"Applied document_types filter: {filters['document_types']}")
            if 'collections' in filters and filters['collections']:
                # Filter by collection names through the many-to-many relationship
                queryset = queryset.filter(
                    document__collection_documents__collection__name__in=filters['collections']
                )
                print(f"Applied collections filter: {filters['collections']}")
            if 'date_range' in filters and filters['date_range']:
                start_date, end_date = filters['date_range']
                queryset = queryset.filter(
                    document__created_at__range=[start_date, end_date]
                )
                print(f"Applied date_range filter: {start_date} to {end_date}")
        print(f" result::: {list(queryset[:top_k])}")
        return list(queryset[:top_k])
    
    def graph_retrieve(
        self,
        query: str,
        max_entities: int = 15,
        max_communities: int = 3,
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Retrieve context using Graph RAG approach."""
        if not self.graph_engine:
            return {'entities': [], 'communities': [], 'relationships': []}
        
        query_embedding = self.embed_text(query)
        return self.graph_engine.generate_graph_context(
            query=query,
            query_embedding=query_embedding,
            max_entities=max_entities,
            max_communities=max_communities,
            include_relationships=include_relationships
        )
    
    def hybrid_retrieve(
        self,
        query: str,
        vector_top_k: int = 10,
        graph_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining vector and graph approaches."""
        if not self.graph_engine:
            # Fall back to vector-only retrieval
            chunks = self.retrieve_context(query, top_k=vector_top_k)
            return [
                {
                    'chunk': chunk,
                    'score': 1.0,  # Placeholder score
                    'source': 'vector'
                }
                for chunk in chunks
            ]
        
        query_embedding = self.embed_text(query)
        return self.graph_engine.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            max_results=vector_top_k
        )
    
    def build_context(
        self,
        chunks: List[DocumentChunk],
        max_tokens: int = 3000
    ) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            # Simple token estimation (4 chars ≈ 1 token)
            chunk_tokens = len(chunk.content) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            context_part = f"[Source: {chunk.document.title}]\n{chunk.content}\n"
            context_parts.append(context_part)
            current_tokens += chunk_tokens
        
        return "\n".join(context_parts)
    
    def build_graph_context(self, graph_data: Dict[str, Any]) -> str:
        """Build context string from graph data."""
        context_parts = []
        
        # Add entity information
        if graph_data.get('entities'):
            context_parts.append("## Relevant Entities:")
            for entity in graph_data['entities'][:10]:  # Limit entities
                context_parts.append(
                    f"- **{entity['name']}** ({entity['type']}): {entity.get('description', 'No description')}"
                )
        
        # Add community information
        if graph_data.get('communities'):
            context_parts.append("\n## Community Context:")
            for community in graph_data['communities']:
                if community.get('summary'):
                    context_parts.append(f"- **{community['name']}**: {community['summary']}")
        
        # Add relationship information
        if graph_data.get('relationships'):
            context_parts.append("\n## Key Relationships:")
            for rel in graph_data['relationships'][:15]:  # Limit relationships
                context_parts.append(
                    f"- {rel['source']} --[{rel['type']}]--> {rel['target']}"
                    + (f": {rel['description']}" if rel.get('description') else "")
                )
        
        return "\n".join(context_parts)
    
    def generate_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate the final prompt for the LLM."""
        prompt_parts = []
        
        # Add system prompt
        if system_prompt or self.session.system_prompt:
            prompt_parts.append(f"System: {system_prompt or self.session.system_prompt}")
        
        # Add conversation history
        if conversation_history:
            prompt_parts.append("## Conversation History:")
            for turn in conversation_history[-5:]:  # Last 5 turns
                prompt_parts.append(f"Human: {turn.get('human', '')}")
                prompt_parts.append(f"Assistant: {turn.get('assistant', '')}")
        
        # Add context
        if context:
            prompt_parts.append(f"## Context Information:\n{context}")
        
        # Add current query
        prompt_parts.append(f"## Current Question:\n{query}")
        
        # Add instruction
        prompt_parts.append(
            "\nPlease provide a comprehensive answer based on the context information above. "
            "If the context doesn't contain sufficient information to answer the question, "
            "please indicate what information is missing."
        )
        
        return "\n\n".join(prompt_parts)
    
    def process_query(
        self,
        query_text: str,
        query_type: str = 'standard',
        retrieval_params: Optional[Dict[str, Any]] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> RAGQuery:
        """Process a complete RAG query."""
        # Create RAG query record
        rag_query = RAGQuery.objects.create(
            session=self.session,
            query_text=query_text,
            query_type=query_type
        )
        
        retrieval_params = retrieval_params or {}
        generation_params = generation_params or {}
        
        try:
            context_chunks = []
            context_text = ""
            
            if query_type == 'standard':
                print("Performing standard vector-based retrieval...")
                # Standard vector-based retrieval
                chunks = self.retrieve_context(
                    query_text,
                    top_k=retrieval_params.get('top_k', 10),
                    similarity_threshold=retrieval_params.get('similarity_threshold', 0.7),
                    filters=retrieval_params.get('filters')
                )
                context_chunks = chunks
                print(f"Retrieved {len(chunks)} chunks for context.")
                context_text = self.build_context(
                    chunks,
                    max_tokens=retrieval_params.get('max_tokens', 3000)
                )
            
            elif query_type == 'graph_rag':
                # Graph-based retrieval
                graph_data = self.graph_retrieve(
                    query_text,
                    max_entities=retrieval_params.get('max_entities', 15),
                    max_communities=retrieval_params.get('max_communities', 3),
                    include_relationships=retrieval_params.get('include_relationships', True)
                )
                context_text = self.build_graph_context(graph_data)
                
                # Also get some vector chunks for completeness
                chunks = self.retrieve_context(query_text, top_k=5)
                context_chunks = chunks
                if chunks:
                    vector_context = self.build_context(chunks, max_tokens=1000)
                    context_text = f"{context_text}\n\n## Additional Context:\n{vector_context}"
            
            elif query_type == 'hybrid':
                # Hybrid retrieval
                hybrid_results = self.hybrid_retrieve(
                    query_text,
                    vector_top_k=retrieval_params.get('top_k', 10),
                    graph_weight=retrieval_params.get('graph_weight', 0.3),
                    vector_weight=retrieval_params.get('vector_weight', 0.7)
                )
                
                context_chunks = [result['chunk'] for result in hybrid_results if 'chunk' in result]
                context_text = self.build_context(
                    context_chunks,
                    max_tokens=retrieval_params.get('max_tokens', 3000)
                )
            
            # Store context relationships
            for i, chunk in enumerate(context_chunks):
                RAGContext.objects.create(
                    query=rag_query,
                    chunk=chunk,
                    relevance_score=1.0,  # Would be computed from similarity
                    rank=i + 1,
                    context_type='retrieval'
                )
            
            # Generate final prompt
            conversation_history = self.session.conversation_context
            final_prompt = self.generate_prompt(
                query_text,
                context_text,
                conversation_history=conversation_history
            )
            
            # Store the prompt and context in metadata
            rag_query.metadata.update({
                'final_prompt': final_prompt,
                'context_text': context_text,
                'context_chunks_count': len(context_chunks),
                'retrieval_params': retrieval_params,
                'generation_params': generation_params
            })
            rag_query.save()
            
            return rag_query
            
        except Exception as e:
            rag_query.metadata['error'] = str(e)
            rag_query.save()
            raise
    
    def update_conversation_context(
        self,
        query: str,
        response: str,
        max_history: int = 10
    ):
        """Update the conversation context for the session."""
        new_turn = {
            'human': query,
            'assistant': response,
            'timestamp': RAGQuery.objects.filter(session=self.session).latest('created_at').created_at.isoformat()
        }
        
        self.session.conversation_context.append(new_turn)
        
        # Keep only the last max_history turns
        if len(self.session.conversation_context) > max_history:
            self.session.conversation_context = self.session.conversation_context[-max_history:]
        
        self.session.save()