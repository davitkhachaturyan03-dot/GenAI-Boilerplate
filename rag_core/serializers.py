from rest_framework import serializers
from .models import RAGSession, RAGQuery, RAGFeedback
from vector_store.models import Document, DocumentChunk, SearchQuery
from graph_rag.models import KnowledgeGraph, Entity, Relationship


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'document_type', 'source', 'created_at', 'metadata']


class DocumentChunkSerializer(serializers.ModelSerializer):
    document = DocumentSerializer(read_only=True)
    
    class Meta:
        model = DocumentChunk
        fields = ['id', 'document', 'content', 'chunk_index', 'token_count', 'metadata']


class KnowledgeGraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeGraph
        fields = ['id', 'name', 'description', 'version', 'statistics', 'created_at']


class RAGSessionSerializer(serializers.ModelSerializer):
    knowledge_graph = KnowledgeGraphSerializer(read_only=True)
    
    class Meta:
        model = RAGSession
        fields = [
            'id', 'session_id', 'user_id', 'knowledge_graph', 
            'conversation_context', 'system_prompt', 'max_context_length',
            'temperature', 'created_at', 'metadata'
        ]


class RAGQuerySerializer(serializers.ModelSerializer):
    session = RAGSessionSerializer(read_only=True)
    context_chunks = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = RAGQuery
        fields = [
            'id', 'session', 'query_text', 'query_type', 'response_text',
            'context_chunks', 'processing_time', 'token_count', 'cost',
            'created_at', 'metadata'
        ]


class RAGFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = RAGFeedback
        fields = ['id', 'query', 'rating', 'feedback_text', 'feedback_type', 'created_at']


class EntitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Entity
        fields = [
            'id', 'name', 'entity_type', 'description', 'aliases',
            'confidence_score', 'frequency', 'created_at'
        ]


class RelationshipSerializer(serializers.ModelSerializer):
    source_entity = EntitySerializer(read_only=True)
    target_entity = EntitySerializer(read_only=True)
    
    class Meta:
        model = Relationship
        fields = [
            'id', 'source_entity', 'target_entity', 'relationship_type',
            'description', 'weight', 'confidence_score', 'frequency', 'created_at'
        ]


class SearchQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchQuery
        fields = [
            'id', 'query_text', 'search_type', 'results_count',
            'filters', 'created_at', 'metadata'
        ]


# Request/Response Serializers
class RAGQueryRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField(max_length=2000)
    query_type = serializers.ChoiceField(
        choices=['standard', 'graph_rag', 'hybrid', 'conversational'],
        default='standard'
    )
    session_id = serializers.CharField(max_length=100, required=False)
    user_id = serializers.CharField(max_length=100, required=False)
    knowledge_graph_id = serializers.IntegerField(required=False)
    
    # Retrieval parameters
    top_k = serializers.IntegerField(default=10, min_value=1, max_value=50)
    similarity_threshold = serializers.FloatField(default=0.7, min_value=0.0, max_value=1.0)
    max_tokens = serializers.IntegerField(default=3000, min_value=100, max_value=8000)
    
    # Generation parameters
    temperature = serializers.FloatField(default=0.7, min_value=0.0, max_value=2.0)
    max_response_tokens = serializers.IntegerField(default=2000, min_value=100, max_value=4000)
    
    # Filters
    document_types = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False,
        allow_empty=True
    )
    collections = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        allow_empty=True
    )
    date_range = serializers.ListField(
        child=serializers.DateTimeField(),
        required=False,
        allow_empty=False,
        min_length=2,
        max_length=2
    )


class VectorSearchRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField(max_length=2000)
    embedding = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        help_text="Pre-computed embedding vector"
    )
    top_k = serializers.IntegerField(default=10, min_value=1, max_value=100)
    distance_metric = serializers.ChoiceField(
        choices=['cosine', 'l2', 'inner_product'],
        default='cosine'
    )
    similarity_threshold = serializers.FloatField(default=0.0, min_value=0.0, max_value=1.0)
    
    # Filters
    document_types = serializers.ListField(
        child=serializers.CharField(max_length=50),
        required=False
    )
    collections = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False
    )
    languages = serializers.ListField(
        child=serializers.CharField(max_length=10),
        required=False
    )


class GraphRAGRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField(max_length=2000)
    knowledge_graph_id = serializers.IntegerField()
    max_entities = serializers.IntegerField(default=15, min_value=1, max_value=50)
    max_communities = serializers.IntegerField(default=3, min_value=1, max_value=10)
    include_relationships = serializers.BooleanField(default=True)
    entity_types = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False
    )


class HybridSearchRequestSerializer(serializers.Serializer):
    query_text = serializers.CharField(max_length=2000)
    top_k = serializers.IntegerField(default=10, min_value=1, max_value=50)
    text_weight = serializers.FloatField(default=0.3, min_value=0.0, max_value=1.0)
    vector_weight = serializers.FloatField(default=0.7, min_value=0.0, max_value=1.0)
    
    def validate(self, data):
        if abs(data['text_weight'] + data['vector_weight'] - 1.0) > 0.01:
            raise serializers.ValidationError(
                "text_weight and vector_weight must sum to 1.0"
            )
        return data


class DocumentUploadSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255)
    content = serializers.CharField()
    document_type = serializers.CharField(max_length=50, default='text')
    source = serializers.URLField(required=False)
    language = serializers.CharField(max_length=10, default='en')
    metadata = serializers.JSONField(default=dict, required=False)
    
    # Chunking parameters
    chunk_size = serializers.IntegerField(default=512, min_value=100, max_value=2000)
    chunk_overlap = serializers.IntegerField(default=50, min_value=0, max_value=500)
    
    # Processing options
    extract_entities = serializers.BooleanField(default=False)
    knowledge_graph_id = serializers.IntegerField(required=False)


class SessionCreateSerializer(serializers.Serializer):
    user_id = serializers.CharField(max_length=100, required=False)
    knowledge_graph_id = serializers.IntegerField(required=False)
    system_prompt = serializers.CharField(required=False)
    max_context_length = serializers.IntegerField(default=4000, min_value=1000, max_value=16000)
    temperature = serializers.FloatField(default=0.7, min_value=0.0, max_value=2.0)


class SearchResultSerializer(serializers.Serializer):
    chunk = DocumentChunkSerializer()
    similarity = serializers.FloatField()
    distance = serializers.FloatField()
    rank = serializers.IntegerField()
    document_title = serializers.CharField()
    document_type = serializers.CharField()


class GraphContextSerializer(serializers.Serializer):
    entities = EntitySerializer(many=True)
    communities = serializers.ListField(child=serializers.DictField())
    relationships = RelationshipSerializer(many=True)
    statistics = serializers.DictField()


class RAGResponseSerializer(serializers.Serializer):
    query_id = serializers.IntegerField()
    session_id = serializers.CharField()
    response_text = serializers.CharField()
    processing_time = serializers.FloatField()
    token_count = serializers.IntegerField()
    context_chunks_count = serializers.IntegerField()
    metadata = serializers.DictField()
    
    # Optional context information
    context_chunks = DocumentChunkSerializer(many=True, required=False)
    graph_context = GraphContextSerializer(required=False)
    search_results = SearchResultSerializer(many=True, required=False)