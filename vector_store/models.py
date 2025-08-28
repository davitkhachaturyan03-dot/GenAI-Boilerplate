from django.db import models
from django.contrib.postgres.fields import ArrayField
from pgvector.django import VectorField
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType


class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        abstract = True


class Document(BaseModel):
    title = models.CharField(max_length=255, db_index=True)
    content = models.TextField()
    source = models.URLField(blank=True, null=True)
    document_type = models.CharField(max_length=50, default='text', db_index=True)
    file_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)
    language = models.CharField(max_length=10, default='en')
    metadata = models.JSONField(default=dict, blank=True)
    
    # Generic foreign key for flexible associations
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, null=True, blank=True)
    object_id = models.PositiveIntegerField(null=True, blank=True)
    content_object = GenericForeignKey('content_type', 'object_id')
    
    class Meta:
        db_table = 'documents'
        indexes = [
            models.Index(fields=['document_type', 'is_active']),
            models.Index(fields=['created_at']),
            models.Index(fields=['file_hash']),
            models.Index(fields=['content_type', 'object_id']),
        ]

    def __str__(self):
        return self.title


class DocumentChunk(BaseModel):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    chunk_index = models.IntegerField()
    start_position = models.IntegerField(default=0)
    end_position = models.IntegerField(default=0)
    token_count = models.IntegerField(default=0)
    embedding = VectorField(dimensions=settings.VECTOR_DIMENSION)
    embedding_model = models.CharField(max_length=100, default=settings.EMBEDDING_MODEL)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'document_chunks'
        unique_together = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['embedding_model']),
            models.Index(fields=['token_count']),
        ]

    def __str__(self):
        return f"{self.document.title} - Chunk {self.chunk_index}"


class VectorIndex(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    dimension = models.IntegerField(default=settings.VECTOR_DIMENSION)
    distance_metric = models.CharField(max_length=20, default='cosine', choices=[
        ('cosine', 'Cosine'),
        ('l2', 'Euclidean'),
        ('inner_product', 'Inner Product'),
    ])
    index_type = models.CharField(max_length=20, default='ivfflat', choices=[
        ('ivfflat', 'IVFFlat'),
        ('hnsw', 'HNSW'),
    ])
    index_parameters = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'vector_indexes'

    def __str__(self):
        return self.name


class EmbeddingModel(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    model_path = models.CharField(max_length=255)
    dimension = models.IntegerField()
    max_sequence_length = models.IntegerField(default=512)
    provider = models.CharField(max_length=50, default='huggingface', choices=[
        ('huggingface', 'Hugging Face'),
        ('openai', 'OpenAI'),
        ('cohere', 'Cohere'),
        ('sentence_transformers', 'Sentence Transformers'),
    ])
    
    class Meta:
        db_table = 'embedding_models'

    def __str__(self):
        return self.name


class SearchQuery(BaseModel):
    query_text = models.TextField()
    query_embedding = VectorField(dimensions=settings.VECTOR_DIMENSION)
    results_count = models.IntegerField(default=0)
    search_type = models.CharField(max_length=20, default='similarity', choices=[
        ('similarity', 'Similarity Search'),
        ('hybrid', 'Hybrid Search'),
        ('semantic', 'Semantic Search'),
        ('graph_rag', 'Graph RAG'),
    ])
    filters = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'search_queries'
        indexes = [
            models.Index(fields=['search_type', 'created_at']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Query: {self.query_text[:50]}..."


class SearchResult(models.Model):
    query = models.ForeignKey(SearchQuery, on_delete=models.CASCADE, related_name='results')
    chunk = models.ForeignKey(DocumentChunk, on_delete=models.CASCADE)
    similarity_score = models.FloatField()
    rank = models.IntegerField()
    rerank_score = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'search_results'
        unique_together = ['query', 'chunk']
        indexes = [
            models.Index(fields=['similarity_score']),
            models.Index(fields=['rank']),
            models.Index(fields=['rerank_score']),
        ]

    def __str__(self):
        return f"Result {self.rank} for {self.query}"


class Collection(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    documents = models.ManyToManyField(Document, through='CollectionDocument')
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'collections'

    def __str__(self):
        return self.name


class CollectionDocument(models.Model):
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)
    weight = models.FloatField(default=1.0)
    
    class Meta:
        db_table = 'collection_documents'
        unique_together = ['collection', 'document']