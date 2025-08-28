import uuid
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.conf import settings

from complete_rag_example import print_setup_instructions
from .models import RAGSession, RAGQuery, RAGFeedback
from .llm_client import llm_client
from .serializers import (
    RAGSessionSerializer, RAGQuerySerializer, RAGFeedbackSerializer,
    RAGQueryRequestSerializer, VectorSearchRequestSerializer,
    GraphRAGRequestSerializer, HybridSearchRequestSerializer,
    DocumentUploadSerializer, SessionCreateSerializer,
    SearchResultSerializer, RAGResponseSerializer
)
from .rag_engine import RAGEngine
from vector_store.models import Document, DocumentChunk
from vector_store.search_engine import VectorSearchEngine
from graph_rag.models import KnowledgeGraph
from graph_rag.graph_rag_engine import GraphRAGEngine
from mcp_integration.models import MCPServer
from mcp_integration.mcp_client import MCPManager


class RAGSessionViewSet(viewsets.ModelViewSet):
    queryset = RAGSession.objects.all()
    serializer_class = RAGSessionSerializer
    
    def create(self, request):
        """Create a new RAG session."""
        serializer = SessionCreateSerializer(data=request.data)
        if serializer.is_valid():
            session_data = serializer.validated_data
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Get knowledge graph if specified
            knowledge_graph = None
            if 'knowledge_graph_id' in session_data:
                try:
                    knowledge_graph = KnowledgeGraph.objects.get(
                        id=session_data['knowledge_graph_id'],
                        is_active=True
                    )
                except KnowledgeGraph.DoesNotExist:
                    return Response(
                        {'error': 'Knowledge graph not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Create session
            session = RAGSession.objects.create(
                session_id=session_id,
                user_id=session_data.get('user_id'),
                knowledge_graph=knowledge_graph,
                system_prompt=session_data.get('system_prompt', ''),
                max_context_length=session_data.get('max_context_length', 4000),
                temperature=session_data.get('temperature', 0.7)
            )
            
            return Response(
                RAGSessionSerializer(session).data,
                status=status.HTTP_201_CREATED
            )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RAGQueryView(APIView):
    """Main RAG query processing endpoint."""
    
    def post(self, request):
        serializer = RAGQueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Get or create session
            session = self._get_or_create_session(data)
            
            # Initialize RAG engine
            rag_engine = RAGEngine(session)
            
            # Process the query
            start_time = timezone.now()
            
            rag_query = rag_engine.process_query(
                query_text=data['query_text'],
                query_type=data['query_type'],
                retrieval_params={
                    'top_k': data['top_k'],
                    'similarity_threshold': data['similarity_threshold'],
                    'max_tokens': data['max_tokens'],
                    'filters': {
                        'document_types': data.get('document_types'),
                        'collections': data.get('collections'),
                        'date_range': data.get('date_range')
                    }
                },
                generation_params={
                    'temperature': data['temperature'],
                    'max_tokens': data['max_response_tokens']
                }
            )
            print(f"RAG Query ID: {rag_query.id}, Query Text: {rag_query.query_text}")
            # Generate response using MCP server (sync call)
            response_text = self._generate_response_sync(rag_query)
            
            # Update query with response
            processing_time = (timezone.now() - start_time).total_seconds()
            rag_query.response_text = response_text
            rag_query.processing_time = processing_time
            rag_query.save()
            
            # Update conversation context
            rag_engine.update_conversation_context(
                data['query_text'], response_text
            )
            
            # Prepare response
            response_data = {
                'query_id': rag_query.id,
                'session_id': session.session_id,
                'response_text': response_text,
                'processing_time': processing_time,
                'token_count': rag_query.token_count,
                'context_chunks_count': rag_query.context_chunks.count(),
                'metadata': rag_query.metadata
            }
            
            # Include context chunks if requested
            if request.query_params.get('include_context') == 'true':
                response_data['context_chunks'] = [
                    {
                        'content': chunk.content,
                        'document_title': chunk.document.title,
                        'similarity': context.relevance_score
                    }
                    for context in rag_query.ragcontext_set.all()
                    for chunk in [context.chunk]
                ]
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Query processing failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _generate_response_sync(self, rag_query: RAGQuery) -> str:
        """Generate response using the unified LLM client (MCP server)."""
        try:
            from .llm_client import generate_rag_response
            
            context_text = rag_query.metadata.get('context_text', '')
            system_prompt = rag_query.session.system_prompt
            print(f"Generating response for RAG Query ID: {rag_query.id} with context length: {len(context_text)}")
            
            response = generate_rag_response(
                query=rag_query.query_text,
                context=context_text,
                system_prompt=system_prompt,
                model=getattr(settings, 'DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')
            )
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _get_or_create_session(self, data) -> RAGSession:
        """Get existing session or create a new one."""
        session_id = data.get('session_id')
        
        if session_id:
            try:
                return RAGSession.objects.get(session_id=session_id, is_active=True)
            except RAGSession.DoesNotExist:
                pass
        
        # Create new session
        knowledge_graph = None
        if data.get('knowledge_graph_id'):
            knowledge_graph = get_object_or_404(
                KnowledgeGraph, 
                id=data['knowledge_graph_id'],
                is_active=True
            )
        
        return RAGSession.objects.create(
            session_id=str(uuid.uuid4()),
            user_id=data.get('user_id'),
            knowledge_graph=knowledge_graph,
            temperature=data['temperature']
        )


class VectorSearchView(APIView):
    """Vector similarity search endpoint."""
    
    def post(self, request):
        serializer = VectorSearchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        print(f"Vector Search Request Data: {data}")
        try:
            # Get or generate embedding
            if 'embedding' in data:
                query_embedding = data['embedding']
            else:
                # Generate embedding using OpenAI via MCP
                embedding_response = llm_client.sync_create_embedding([data['query_text']])
                if 'error' in embedding_response:
                    return Response(
                        {'error': f"Embedding generation failed: {embedding_response['error']['message']}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                query_embedding = embedding_response['data'][0]['embedding']
            print(f"Generated Query Embedding: {query_embedding[:5]}... (truncated)")
            # Initialize search engine
            search_engine = VectorSearchEngine()
            
            # Perform search
            results = search_engine.semantic_search(
                query_embedding=query_embedding,
                top_k=data['top_k'],
                distance_metric=data['distance_metric'],
                similarity_threshold=data['similarity_threshold'],
                filters={
                    'document_types': data.get('document_types'),
                    'collections': data.get('collections'),
                    'languages': data.get('languages')
                }
            )
            
            # Create search query record
            search_query = search_engine.create_search_query_record(
                query_text=data['query_text'],
                query_embedding=query_embedding,
                search_type='similarity',
                results=results
            )
            
            # Format response
            response_data = {
                'search_query_id': search_query.id,
                'results_count': len(results),
                'results': [
                    {
                        'chunk_id': result['chunk'].id,
                        'similarity': result['similarity'],
                        'distance': result['distance'],
                        'content': result['content'],
                        'document_title': result['document_title'],
                        'document_type': result['document_type'],
                        'metadata': result['metadata']
                    }
                    for result in results
                ]
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Vector search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GraphRAGView(APIView):
    """Graph RAG query endpoint."""
    
    def post(self, request):
        serializer = GraphRAGRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Get knowledge graph
            knowledge_graph = get_object_or_404(
                KnowledgeGraph,
                id=data['knowledge_graph_id'],
                is_active=True
            )
            
            # Initialize Graph RAG engine
            graph_engine = GraphRAGEngine(knowledge_graph)
            
            # Generate embeddings using OpenAI via MCP
            embedding_response = llm_client.sync_create_embedding([data['query_text']])
            if 'error' in embedding_response:
                return Response(
                    {'error': f"Embedding generation failed: {embedding_response['error']['message']}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            query_embedding = embedding_response['data'][0]['embedding']
            
            # Generate graph context
            graph_context = graph_engine.generate_graph_context(
                query=data['query_text'],
                query_embedding=query_embedding,
                max_entities=data['max_entities'],
                max_communities=data['max_communities'],
                include_relationships=data['include_relationships']
            )
            
            # Perform hybrid search if requested
            hybrid_results = graph_engine.hybrid_search(
                query=data['query_text'],
                query_embedding=query_embedding,
                max_results=data['max_entities']
            )
            
            response_data = {
                'knowledge_graph_id': knowledge_graph.id,
                'query_text': data['query_text'],
                'graph_context': graph_context,
                'hybrid_results': [
                    {
                        'content': result.get('content', ''),
                        'vector_score': result.get('vector_score', 0.0),
                        'graph_score': result.get('graph_score', 0.0),
                        'hybrid_score': result.get('hybrid_score', 0.0),
                        'document_title': result.get('document_title', '')
                    }
                    for result in hybrid_results
                ]
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Graph RAG failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HybridSearchView(APIView):
    """Hybrid search combining vector and text search."""
    
    def post(self, request):
        serializer = HybridSearchRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            # Initialize search engine
            search_engine = VectorSearchEngine()
            
            # Generate embedding using OpenAI via MCP
            embedding_response = llm_client.sync_create_embedding([data['query_text']])
            if 'error' in embedding_response:
                return Response(
                    {'error': f"Embedding generation failed: {embedding_response['error']['message']}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            query_embedding = embedding_response['data'][0]['embedding']
            
            # Perform hybrid search
            results = search_engine.hybrid_search(
                query_text=data['query_text'],
                query_embedding=query_embedding,
                top_k=data['top_k'],
                text_weight=data['text_weight'],
                vector_weight=data['vector_weight']
            )
            
            response_data = {
                'results_count': len(results),
                'text_weight': data['text_weight'],
                'vector_weight': data['vector_weight'],
                'results': [
                    {
                        'content': result['content'],
                        'document_title': result['document_title'],
                        'vector_score': result['vector_score'],
                        'text_score': result['text_score'],
                        'hybrid_score': result['hybrid_score']
                    }
                    for result in results
                ]
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Hybrid search failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DocumentUploadView(APIView):
    """Document upload and processing endpoint."""
    
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        
        try:
            print("Starting document upload and processing...")
            # Create document
            document = Document.objects.create(
                title=data['title'],
                content=data['content'],
                document_type=data['document_type'],
                source=data.get('source'),
                language=data['language'],
                metadata=data.get('metadata', {})
            )
            print(f"Created document with ID: {document.id}")
            
            # Process and create chunks
            chunks_created = self._create_document_chunks(
                document=document,
                content=data['content'],
                chunk_size=data['chunk_size'],
                chunk_overlap=data['chunk_overlap']
            )
            print(f"Created {chunks_created} chunks for document ID: {document.id}")
            
            # Extract entities if requested
            entities_extracted = 0
            if data.get('extract_entities') and data.get('knowledge_graph_id'):
                entities_extracted = self._extract_entities(
                    document=document,
                    knowledge_graph_id=data['knowledge_graph_id']
                )
            print(f"Extracted {entities_extracted} entities for document ID: {document.id}")
            response_data = {
                'document_id': document.id,
                'title': document.title,
                'chunks_created': chunks_created,
                'entities_extracted': entities_extracted,
                'created_at': document.created_at
            }
            print(f"Document upload response: {response_data}")
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': f'Document upload failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _create_document_chunks(self, document, content, chunk_size, chunk_overlap):
        """Create document chunks with embeddings."""
        
        # Simple text chunking (can be enhanced with more sophisticated methods)
        chunks = []
        start = 0
        chunk_index = 0
        chunk_texts = []
        
        # First, create all chunks and collect their texts
        while start < len(content):
            end = start + chunk_size
            chunk_content = content[start:end]
            chunk_texts.append(chunk_content)
            start = end - chunk_overlap
        
        # Generate embeddings for all chunks at once using OpenAI via MCP
        try:
            print("Generating embeddings for document chunks...")
            embedding_response = llm_client.sync_create_embedding(chunk_texts)
            
            if 'error' in embedding_response:
                print(f"Embedding generation error: {embedding_response['error']['message']}")
                raise Exception(f"Embedding generation failed: {embedding_response['error']['message']}")

            
            embeddings = [item['embedding'] for item in embedding_response['data']]
        except Exception as e:
            print(f"Exception during embedding generation: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
        
        # Create chunks with their embeddings
        start = 0
        for chunk_index, (chunk_content, embedding) in enumerate(zip(chunk_texts, embeddings)):
            end = start + chunk_size
            
            chunk = DocumentChunk.objects.create(
                document=document,
                content=chunk_content,
                chunk_index=chunk_index,
                start_position=start,
                end_position=min(end, len(content)),
                token_count=len(chunk_content.split()),
                embedding=embedding,
                embedding_model='text-embedding-ada-002'
            )
            
            chunks.append(chunk)
            start = end - chunk_overlap
        
        return len(chunks)
    
    def _extract_entities(self, document, knowledge_graph_id):
        """Extract entities from document (placeholder implementation)."""
        # This would implement NER and entity extraction
        # For now, return 0 as placeholder
        return 0


class RAGFeedbackView(APIView):
    """Endpoint for collecting RAG query feedback."""
    
    def post(self, request):
        serializer = RAGFeedbackSerializer(data=request.data)
        if serializer.is_valid():
            feedback = serializer.save()
            return Response(
                RAGFeedbackSerializer(feedback).data,
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)