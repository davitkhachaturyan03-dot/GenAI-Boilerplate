import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from django.db.models import Q, F
from django.conf import settings
from pgvector.django import CosineDistance

from .models import Entity, Relationship, Community, KnowledgeGraph, GraphQuery
from vector_store.models import DocumentChunk, SearchQuery


class GraphRAGEngine:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.graph = None
        self._load_graph()
    
    def _load_graph(self):
        """Load the knowledge graph into NetworkX format for efficient querying."""
        self.graph = nx.DiGraph()
        
        # Add entities as nodes
        entities = Entity.objects.filter(
            knowledgegraph=self.knowledge_graph,
            is_active=True
        ).select_related()
        
        for entity in entities:
            self.graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                embedding=entity.embedding,
                confidence=entity.confidence_score,
                metadata=entity.metadata
            )
        
        # Add relationships as edges
        relationships = Relationship.objects.filter(
            source_entity__knowledgegraph=self.knowledge_graph,
            is_active=True
        ).select_related('source_entity', 'target_entity')
        
        for rel in relationships:
            self.graph.add_edge(
                rel.source_entity.id,
                rel.target_entity.id,
                relationship_type=rel.relationship_type,
                weight=rel.weight,
                confidence=rel.confidence_score,
                metadata=rel.metadata
            )
    
    def find_relevant_entities(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        entity_types: Optional[List[str]] = None
    ) -> List[Entity]:
        """Find entities most similar to the query embedding."""
        queryset = Entity.objects.filter(
            knowledgegraph=self.knowledge_graph,
            is_active=True
        ).annotate(
            distance=CosineDistance('embedding', query_embedding)
        ).order_by('distance')
        
        if entity_types:
            queryset = queryset.filter(entity_type__in=entity_types)
        
        return list(queryset[:top_k])
    
    def find_relevant_communities(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        level: Optional[int] = None
    ) -> List[Community]:
        """Find communities most relevant to the query."""
        queryset = Community.objects.filter(
            knowledge_graph=self.knowledge_graph,
            is_active=True,
            summary_embedding__isnull=False
        ).annotate(
            distance=CosineDistance('summary_embedding', query_embedding)
        ).order_by('distance')
        
        if level is not None:
            queryset = queryset.filter(level=level)
        
        return list(queryset[:top_k])
    
    def get_entity_neighborhood(
        self, 
        entity_ids: List[int], 
        hop_distance: int = 2,
        max_nodes: int = 50
    ) -> Dict[str, Any]:
        """Get the neighborhood subgraph around specified entities."""
        if not self.graph:
            self._load_graph()
        
        neighborhood_nodes = set(entity_ids)
        
        for hop in range(hop_distance):
            current_level_nodes = set()
            for node in neighborhood_nodes:
                if node in self.graph:
                    # Add neighbors (both incoming and outgoing)
                    current_level_nodes.update(self.graph.successors(node))
                    current_level_nodes.update(self.graph.predecessors(node))
            
            neighborhood_nodes.update(current_level_nodes)
            
            if len(neighborhood_nodes) > max_nodes:
                # Prioritize by node centrality if we have too many nodes
                centrality = nx.degree_centrality(self.graph.subgraph(neighborhood_nodes))
                sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                neighborhood_nodes = set([node for node, _ in sorted_nodes[:max_nodes]])
                break
        
        subgraph = self.graph.subgraph(neighborhood_nodes)
        
        return {
            'nodes': list(subgraph.nodes(data=True)),
            'edges': list(subgraph.edges(data=True)),
            'node_count': len(subgraph.nodes),
            'edge_count': len(subgraph.edges)
        }
    
    def compute_node_importance(
        self, 
        nodes: List[int],
        algorithm: str = 'pagerank'
    ) -> Dict[int, float]:
        """Compute importance scores for nodes using various graph algorithms."""
        if not self.graph:
            self._load_graph()
        
        subgraph = self.graph.subgraph(nodes) if nodes else self.graph
        
        if algorithm == 'pagerank':
            return nx.pagerank(subgraph, weight='weight')
        elif algorithm == 'betweenness':
            return nx.betweenness_centrality(subgraph, weight='weight')
        elif algorithm == 'closeness':
            return nx.closeness_centrality(subgraph, distance='weight')
        elif algorithm == 'degree':
            return nx.degree_centrality(subgraph)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def generate_graph_context(
        self, 
        query: str,
        query_embedding: np.ndarray,
        max_entities: int = 20,
        max_communities: int = 5,
        include_relationships: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive context for Graph RAG."""
        
        # Find relevant entities
        relevant_entities = self.find_relevant_entities(
            query_embedding, 
            top_k=max_entities
        )
        entity_ids = [e.id for e in relevant_entities]
        
        # Find relevant communities
        relevant_communities = self.find_relevant_communities(
            query_embedding,
            top_k=max_communities
        )
        
        # Get neighborhood context
        neighborhood = self.get_entity_neighborhood(entity_ids)
        
        # Compute importance scores
        importance_scores = self.compute_node_importance(entity_ids)
        
        # Get relationship context if requested
        relationship_context = []
        if include_relationships and entity_ids:
            relationships = Relationship.objects.filter(
                Q(source_entity__id__in=entity_ids) | 
                Q(target_entity__id__in=entity_ids),
                is_active=True
            ).select_related('source_entity', 'target_entity').order_by('-confidence_score')[:50]
            
            relationship_context = [
                {
                    'source': rel.source_entity.name,
                    'target': rel.target_entity.name,
                    'type': rel.relationship_type,
                    'description': rel.description,
                    'confidence': rel.confidence_score
                }
                for rel in relationships
            ]
        
        return {
            'entities': [
                {
                    'id': e.id,
                    'name': e.name,
                    'type': e.entity_type,
                    'description': e.description,
                    'confidence': e.confidence_score,
                    'importance': importance_scores.get(e.id, 0.0)
                }
                for e in relevant_entities
            ],
            'communities': [
                {
                    'id': c.id,
                    'name': c.name,
                    'summary': c.summary,
                    'level': c.level,
                    'size': c.size,
                    'density': c.density
                }
                for c in relevant_communities
            ],
            'relationships': relationship_context,
            'neighborhood': neighborhood,
            'query': query,
            'statistics': {
                'total_entities': len(relevant_entities),
                'total_communities': len(relevant_communities),
                'total_relationships': len(relationship_context),
                'neighborhood_nodes': neighborhood['node_count'],
                'neighborhood_edges': neighborhood['edge_count']
            }
        }
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and graph structure."""
        
        # Vector-based search
        vector_results = DocumentChunk.objects.filter(
            document__is_active=True,
            is_active=True
        ).annotate(
            vector_distance=CosineDistance('embedding', query_embedding)
        ).order_by('vector_distance')[:max_results * 2]
        
        # Graph-based context
        graph_context = self.generate_graph_context(query, query_embedding)
        relevant_entity_names = [e['name'].lower() for e in graph_context['entities']]
        
        # Combine and re-rank results
        hybrid_results = []
        for chunk in vector_results:
            vector_score = 1 - chunk.vector_distance  # Convert distance to similarity
            
            # Calculate graph relevance score
            graph_score = 0.0
            chunk_content_lower = chunk.content.lower()
            for entity_name in relevant_entity_names:
                if entity_name in chunk_content_lower:
                    graph_score += 1.0
            
            graph_score = min(graph_score / len(relevant_entity_names), 1.0) if relevant_entity_names else 0.0
            
            # Combine scores
            hybrid_score = (vector_weight * vector_score) + (graph_weight * graph_score)
            
            hybrid_results.append({
                'chunk': chunk,
                'vector_score': vector_score,
                'graph_score': graph_score,
                'hybrid_score': hybrid_score,
                'document_title': chunk.document.title,
                'content': chunk.content
            })
        
        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_results[:max_results]
    
    def find_shortest_paths(
        self,
        source_entity_id: int,
        target_entity_id: int,
        max_paths: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """Find shortest paths between two entities in the knowledge graph."""
        if not self.graph:
            self._load_graph()
        
        try:
            paths = list(nx.all_shortest_paths(
                self.graph, 
                source_entity_id, 
                target_entity_id,
                weight='weight'
            ))[:max_paths]
            
            result_paths = []
            for path in paths:
                path_info = []
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i + 1]]
                    path_info.append({
                        'source_id': path[i],
                        'target_id': path[i + 1],
                        'relationship_type': edge_data.get('relationship_type', 'unknown'),
                        'weight': edge_data.get('weight', 1.0),
                        'confidence': edge_data.get('confidence', 0.0)
                    })
                result_paths.append(path_info)
            
            return result_paths
        except nx.NetworkXNoPath:
            return []