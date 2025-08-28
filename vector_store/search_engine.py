import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from django.db.models import Q, F
from django.db import connection
from pgvector.django import CosineDistance, L2Distance

from .models import DocumentChunk, SearchQuery, SearchResult, VectorIndex, Document


class VectorSearchEngine:
    """Advanced vector search engine with multiple similarity metrics and optimization."""
    
    def __init__(self):
        self.supported_distances = {
            'cosine': CosineDistance,
            'l2': L2Distance,
        }
    
    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        distance_metric: str = 'cosine',
        similarity_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        print("Starting semantic search...")
        distance_func = self.supported_distances.get(distance_metric, CosineDistance)
        
        queryset = DocumentChunk.objects.filter(
            is_active=True,
            document__is_active=True
        ).select_related('document').annotate(
            distance=distance_func('embedding', query_embedding)
        )
        print("Annotated distances...")
        # Apply similarity threshold
        if distance_metric == 'cosine':
            # For cosine distance, lower is better, convert to similarity
            queryset = queryset.annotate(
                similarity=1 - F('distance')
            ).filter(similarity__gte=similarity_threshold)
        else:
            queryset = queryset.filter(distance__lte=1 - similarity_threshold)
        print("Applied similarity threshold...")
        # Apply filters
        print(f"BEFORE FILTERS: {queryset.count()} results")

        if filters:
            queryset = self._apply_filters(queryset, filters)

        print(f"AFTER FILTERS: {queryset.count()} results")
        # Order and limit results
        queryset = queryset.order_by('distance')[:top_k]
        
        results = []
        for chunk in queryset:
            similarity = 1 - chunk.distance if distance_metric == 'cosine' else chunk.distance
            results.append({
                'chunk': chunk,
                'similarity': similarity,
                'distance': chunk.distance,
                'document_title': chunk.document.title,
                'document_type': chunk.document.document_type,
                'content': chunk.content,
                'metadata': chunk.metadata
            })
        print("Completed semantic search END OF VIEW.")
        return results
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector similarity."""
        
        # Vector search
        vector_results = self.semantic_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results to rerank
            filters=filters
        )
        
        # Text search using PostgreSQL full-text search
        text_results = self._full_text_search(query_text, top_k * 2, filters)
        
        # Combine and rerank results
        combined_results = self._combine_search_results(
            vector_results, text_results, text_weight, vector_weight
        )
        
        return sorted(combined_results, key=lambda x: x['hybrid_score'], reverse=True)[:top_k]
    
    def multi_vector_search(
        self,
        query_embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
        aggregation_method: str = 'weighted_avg'
    ) -> List[Dict[str, Any]]:
        """Search using multiple query embeddings with different aggregation methods."""
        
        if not query_embeddings:
            return []
        
        weights = weights or [1.0] * len(query_embeddings)
        if len(weights) != len(query_embeddings):
            weights = [1.0] * len(query_embeddings)
        
        all_results = {}  # chunk_id -> {chunk, scores}
        
        for i, embedding in enumerate(query_embeddings):
            results = self.semantic_search(embedding, top_k=top_k * 2)
            
            for result in results:
                chunk_id = result['chunk'].id
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        'chunk': result['chunk'],
                        'scores': [],
                        'similarities': []
                    }
                
                all_results[chunk_id]['scores'].append(result['similarity'] * weights[i])
                all_results[chunk_id]['similarities'].append(result['similarity'])
        
        # Aggregate scores
        final_results = []
        for chunk_id, data in all_results.items():
            if aggregation_method == 'weighted_avg':
                final_score = sum(data['scores']) / len(data['scores'])
            elif aggregation_method == 'max':
                final_score = max(data['scores'])
            elif aggregation_method == 'sum':
                final_score = sum(data['scores'])
            else:
                final_score = sum(data['scores']) / len(data['scores'])
            
            final_results.append({
                'chunk': data['chunk'],
                'similarity': final_score,
                'individual_similarities': data['similarities'],
                'aggregation_method': aggregation_method,
                'content': data['chunk'].content,
                'document_title': data['chunk'].document.title
            })
        
        return sorted(final_results, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def approximate_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        index_name: str,
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Use approximate nearest neighbor search for large-scale retrieval."""
        
        search_params = search_params or {}
        
        # Use PostgreSQL's vector index for ANN search
        with connection.cursor() as cursor:
            # This would use pgvector's IVFFlat or HNSW index
            cursor.execute("""
                SELECT 
                    dc.id,
                    dc.content,
                    dc.embedding <=> %s AS distance,
                    d.title,
                    d.document_type
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.is_active = true AND d.is_active = true
                ORDER BY dc.embedding <=> %s
                LIMIT %s
            """, [query_embedding.tolist(), query_embedding.tolist(), top_k])
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, distance, title, doc_type = row
                results.append({
                    'chunk_id': chunk_id,
                    'content': content,
                    'distance': distance,
                    'similarity': 1 - distance,
                    'document_title': title,
                    'document_type': doc_type
                })
        
        return results
    
    def batch_search(
        self,
        queries: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform batch search for multiple queries efficiently."""
        
        results = {}
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            for query in batch:
                query_id = query.get('id', f'query_{i}')
                query_embedding = query['embedding']
                search_params = query.get('params', {})
                
                search_results = self.semantic_search(
                    query_embedding=query_embedding,
                    top_k=search_params.get('top_k', 10),
                    distance_metric=search_params.get('distance_metric', 'cosine'),
                    filters=search_params.get('filters')
                )
                
                results[query_id] = search_results
        
        return results
    
    def create_search_query_record(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        search_type: str = 'similarity',
        results: Optional[List[Dict[str, Any]]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchQuery:
        """Create a search query record for analytics and caching."""
        
        search_query = SearchQuery.objects.create(
            query_text=query_text,
            query_embedding=query_embedding,
            search_type=search_type,
            results_count=len(results) if results else 0,
            filters=filters or {},
            metadata={
                'embedding_dimension': len(query_embedding),
                'search_timestamp': SearchQuery.objects.model._meta.get_field('created_at').auto_now_add
            }
        )
        
        # Create search result records
        if results:
            for i, result in enumerate(results):
                SearchResult.objects.create(
                    query=search_query,
                    chunk=result['chunk'],
                    similarity_score=result['similarity'],
                    rank=i + 1,
                    metadata={
                        'distance': result.get('distance', 0.0),
                        'document_type': result.get('document_type', ''),
                        'search_method': search_type
                    }
                )
        
        return search_query
    
    def get_similar_documents(
        self,
        document: Document,
        top_k: int = 10,
        exclude_same_document: bool = True
    ) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        
        # Get document embeddings (average of chunk embeddings)
        chunks = document.chunks.filter(is_active=True)
        if not chunks.exists():
            return []
        
        # Calculate document-level embedding as average of chunk embeddings
        embeddings = [chunk.embedding for chunk in chunks]
        if not embeddings:
            return []
        
        doc_embedding = np.mean(embeddings, axis=0)
        
        # Search for similar chunks
        results = self.semantic_search(
            query_embedding=doc_embedding,
            top_k=top_k * 3  # Get more to filter out same document
        )
        
        # Group by document and calculate document-level similarity
        doc_similarities = {}
        for result in results:
            result_doc = result['chunk'].document
            
            if exclude_same_document and result_doc.id == document.id:
                continue
            
            if result_doc.id not in doc_similarities:
                doc_similarities[result_doc.id] = {
                    'document': result_doc,
                    'similarities': [],
                    'chunks': []
                }
            
            doc_similarities[result_doc.id]['similarities'].append(result['similarity'])
            doc_similarities[result_doc.id]['chunks'].append(result['chunk'])
        
        # Calculate average similarity per document
        similar_docs = []
        for doc_id, data in doc_similarities.items():
            avg_similarity = np.mean(data['similarities'])
            similar_docs.append({
                'document': data['document'],
                'similarity': avg_similarity,
                'matching_chunks': len(data['chunks']),
                'top_chunks': data['chunks'][:3]  # Top 3 most similar chunks
            })
        
        return sorted(similar_docs, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    def _apply_filters(self, queryset, filters: Dict[str, Any]):
        """Apply various filters to the queryset."""
        
        if 'document_types' in filters and filters['document_types']:
            queryset = queryset.filter(document__document_type__in=filters['document_types'])
        
        if 'collections' in filters and filters['collections']:
            queryset = queryset.filter(document__collection__name__in=filters['collections'])
        
        if 'languages' in filters and filters['languages']:
            queryset = queryset.filter(document__language__in=filters['languages'])
        
        if 'date_range' in filters and filters['date_range']:
            start_date, end_date = filters['date_range']
            queryset = queryset.filter(document__created_at__range=[start_date, end_date])
        
        if 'min_token_count' in filters and filters['min_token_count']:
            queryset = queryset.filter(token_count__gte=filters['min_token_count'])
        
        if 'max_token_count' in filters and filters['max_token_count']:
            queryset = queryset.filter(token_count__lte=filters['max_token_count'])
        
        return queryset
    
    def _full_text_search(self, query_text: str, top_k: int, filters: Optional[Dict[str, Any]] = None):
        """Perform full-text search using PostgreSQL."""
        
        queryset = DocumentChunk.objects.filter(
            is_active=True,
            document__is_active=True
        ).extra(
            select={
                'rank': "ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s))"
            },
            select_params=[query_text],
            where=["to_tsvector('english', content) @@ plainto_tsquery('english', %s)"],
            params=[query_text],
            order_by=['-rank']
        ).select_related('document')
        
        if filters:
            queryset = self._apply_filters(queryset, filters)
        
        results = []
        for chunk in queryset[:top_k]:
            results.append({
                'chunk': chunk,
                'text_score': chunk.rank,
                'content': chunk.content,
                'document_title': chunk.document.title
            })
        
        return results
    
    def _combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        text_weight: float,
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine vector and text search results."""
        
        # Normalize scores
        if vector_results:
            max_vector_score = max(r['similarity'] for r in vector_results)
            for result in vector_results:
                result['normalized_vector_score'] = result['similarity'] / max_vector_score
        
        if text_results:
            max_text_score = max(r['text_score'] for r in text_results)
            for result in text_results:
                result['normalized_text_score'] = result['text_score'] / max_text_score
        
        # Combine results
        combined = {}
        
        for result in vector_results:
            chunk_id = result['chunk'].id
            combined[chunk_id] = {
                'chunk': result['chunk'],
                'vector_score': result['normalized_vector_score'],
                'text_score': 0.0,
                'document_title': result['document_title'],
                'content': result['content']
            }
        
        for result in text_results:
            chunk_id = result['chunk'].id
            if chunk_id in combined:
                combined[chunk_id]['text_score'] = result['normalized_text_score']
            else:
                combined[chunk_id] = {
                    'chunk': result['chunk'],
                    'vector_score': 0.0,
                    'text_score': result['normalized_text_score'],
                    'document_title': result['document_title'],
                    'content': result['content']
                }
        
        # Calculate hybrid scores
        final_results = []
        for chunk_id, data in combined.items():
            hybrid_score = (vector_weight * data['vector_score']) + (text_weight * data['text_score'])
            data['hybrid_score'] = hybrid_score
            final_results.append(data)
        
        return final_results