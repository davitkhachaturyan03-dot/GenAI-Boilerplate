from django.db import models
from django.contrib.postgres.fields import ArrayField
from pgvector.django import VectorField
from django.conf import settings
from vector_store.models import BaseModel, Document, DocumentChunk


class Entity(BaseModel):
    name = models.CharField(max_length=255, db_index=True)
    entity_type = models.CharField(max_length=100, db_index=True)
    description = models.TextField(blank=True)
    aliases = ArrayField(models.CharField(max_length=255), default=list, blank=True)
    embedding = VectorField(dimensions=settings.VECTOR_DIMENSION)
    confidence_score = models.FloatField(default=0.0)
    frequency = models.IntegerField(default=1)
    source_documents = models.ManyToManyField(Document, through='EntityDocument')
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'entities'
        unique_together = ['name', 'entity_type']
        indexes = [
            models.Index(fields=['entity_type', 'confidence_score']),
            models.Index(fields=['frequency']),
            models.Index(fields=['name', 'entity_type']),
        ]

    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class Relationship(BaseModel):
    source_entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name='outgoing_relationships')
    target_entity = models.ForeignKey(Entity, on_delete=models.CASCADE, related_name='incoming_relationships')
    relationship_type = models.CharField(max_length=100, db_index=True)
    description = models.TextField(blank=True)
    weight = models.FloatField(default=1.0)
    confidence_score = models.FloatField(default=0.0)
    frequency = models.IntegerField(default=1)
    source_chunks = models.ManyToManyField(DocumentChunk, through='RelationshipChunk')
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'relationships'
        unique_together = ['source_entity', 'target_entity', 'relationship_type']
        indexes = [
            models.Index(fields=['relationship_type', 'weight']),
            models.Index(fields=['confidence_score']),
            models.Index(fields=['frequency']),
        ]

    def __str__(self):
        return f"{self.source_entity.name} -> {self.relationship_type} -> {self.target_entity.name}"


class KnowledgeGraph(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    entities = models.ManyToManyField(Entity, blank=True)
    version = models.CharField(max_length=20, default='1.0')
    schema_version = models.CharField(max_length=20, default='1.0')
    statistics = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'knowledge_graphs'

    def __str__(self):
        return self.name


class Community(BaseModel):
    name = models.CharField(max_length=255)
    knowledge_graph = models.ForeignKey(KnowledgeGraph, on_delete=models.CASCADE, related_name='communities')
    entities = models.ManyToManyField(Entity, related_name='communities')
    level = models.IntegerField(default=0)
    size = models.IntegerField(default=0)
    density = models.FloatField(default=0.0)
    modularity_score = models.FloatField(default=0.0)
    summary = models.TextField(blank=True)
    summary_embedding = VectorField(dimensions=settings.VECTOR_DIMENSION, null=True, blank=True)
    parent_community = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='sub_communities')
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'communities'
        unique_together = ['knowledge_graph', 'name', 'level']
        indexes = [
            models.Index(fields=['knowledge_graph', 'level']),
            models.Index(fields=['size', 'density']),
            models.Index(fields=['modularity_score']),
        ]

    def __str__(self):
        return f"{self.name} (Level {self.level})"


class GraphQuery(BaseModel):
    query_text = models.TextField()
    query_type = models.CharField(max_length=50, default='graph_rag', choices=[
        ('graph_rag', 'Graph RAG'),
        ('entity_search', 'Entity Search'),
        ('relationship_search', 'Relationship Search'),
        ('community_search', 'Community Search'),
        ('path_search', 'Path Search'),
    ])
    knowledge_graph = models.ForeignKey(KnowledgeGraph, on_delete=models.CASCADE)
    query_embedding = VectorField(dimensions=settings.VECTOR_DIMENSION)
    parameters = models.JSONField(default=dict, blank=True)
    results_count = models.IntegerField(default=0)
    execution_time = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'graph_queries'
        indexes = [
            models.Index(fields=['query_type', 'created_at']),
            models.Index(fields=['knowledge_graph', 'created_at']),
        ]

    def __str__(self):
        return f"Graph Query: {self.query_text[:50]}..."


class EntityDocument(models.Model):
    entity = models.ForeignKey(Entity, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunks = models.ManyToManyField(DocumentChunk, blank=True)
    confidence_score = models.FloatField(default=0.0)
    extraction_method = models.CharField(max_length=50, default='ner')
    
    class Meta:
        db_table = 'entity_documents'
        unique_together = ['entity', 'document']


class RelationshipChunk(models.Model):
    relationship = models.ForeignKey(Relationship, on_delete=models.CASCADE)
    chunk = models.ForeignKey(DocumentChunk, on_delete=models.CASCADE)
    confidence_score = models.FloatField(default=0.0)
    extraction_method = models.CharField(max_length=50, default='relation_extraction')
    
    class Meta:
        db_table = 'relationship_chunks'
        unique_together = ['relationship', 'chunk']


class GraphIndex(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    knowledge_graph = models.ForeignKey(KnowledgeGraph, on_delete=models.CASCADE, related_name='indexes')
    index_type = models.CharField(max_length=50, choices=[
        ('entity_embedding', 'Entity Embedding Index'),
        ('community_embedding', 'Community Embedding Index'),
        ('relationship_weight', 'Relationship Weight Index'),
        ('hybrid', 'Hybrid Index'),
    ])
    parameters = models.JSONField(default=dict, blank=True)
    statistics = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'graph_indexes'

    def __str__(self):
        return f"{self.name} ({self.index_type})"


class TemporalEdge(BaseModel):
    relationship = models.ForeignKey(Relationship, on_delete=models.CASCADE, related_name='temporal_edges')
    timestamp = models.DateTimeField()
    weight = models.FloatField(default=1.0)
    event_type = models.CharField(max_length=50, default='occurrence')
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'temporal_edges'
        indexes = [
            models.Index(fields=['relationship', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['event_type']),
        ]

    def __str__(self):
        return f"{self.relationship} at {self.timestamp}"