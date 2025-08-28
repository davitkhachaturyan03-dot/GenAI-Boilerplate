from django.db import models
from vector_store.models import BaseModel, Document, DocumentChunk
from graph_rag.models import KnowledgeGraph


class RAGSession(BaseModel):
    session_id = models.CharField(max_length=100, unique=True)
    user_id = models.CharField(max_length=100, null=True, blank=True)
    knowledge_graph = models.ForeignKey(KnowledgeGraph, on_delete=models.CASCADE, null=True, blank=True)
    conversation_context = models.JSONField(default=list, blank=True)
    system_prompt = models.TextField(blank=True)
    max_context_length = models.IntegerField(default=4000)
    temperature = models.FloatField(default=0.7)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'rag_sessions'
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['user_id', 'created_at']),
        ]

    def __str__(self):
        return f"RAG Session: {self.session_id}"


class RAGQuery(BaseModel):
    session = models.ForeignKey(RAGSession, on_delete=models.CASCADE, related_name='queries')
    query_text = models.TextField()
    query_type = models.CharField(max_length=50, default='standard', choices=[
        ('standard', 'Standard RAG'),
        ('graph_rag', 'Graph RAG'),
        ('hybrid', 'Hybrid RAG'),
        ('conversational', 'Conversational RAG'),
    ])
    response_text = models.TextField(blank=True)
    context_chunks = models.ManyToManyField(DocumentChunk, through='RAGContext')
    processing_time = models.FloatField(null=True, blank=True)
    token_count = models.IntegerField(default=0)
    cost = models.DecimalField(max_digits=10, decimal_places=6, default=0.0)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        db_table = 'rag_queries'
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['query_type']),
            models.Index(fields=['processing_time']),
        ]

    def __str__(self):
        return f"Query: {self.query_text[:50]}..."


class RAGContext(models.Model):
    query = models.ForeignKey(RAGQuery, on_delete=models.CASCADE)
    chunk = models.ForeignKey(DocumentChunk, on_delete=models.CASCADE)
    relevance_score = models.FloatField()
    rank = models.IntegerField()
    context_type = models.CharField(max_length=20, default='retrieval', choices=[
        ('retrieval', 'Retrieved Context'),
        ('graph', 'Graph Context'),
        ('conversation', 'Conversation Context'),
    ])
    
    class Meta:
        db_table = 'rag_contexts'
        unique_together = ['query', 'chunk']
        indexes = [
            models.Index(fields=['relevance_score']),
            models.Index(fields=['rank']),
            models.Index(fields=['context_type']),
        ]


class RAGFeedback(BaseModel):
    query = models.ForeignKey(RAGQuery, on_delete=models.CASCADE, related_name='feedback')
    user_id = models.CharField(max_length=100, null=True, blank=True)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    feedback_text = models.TextField(blank=True)
    feedback_type = models.CharField(max_length=20, default='quality', choices=[
        ('quality', 'Response Quality'),
        ('relevance', 'Content Relevance'),
        ('accuracy', 'Factual Accuracy'),
        ('completeness', 'Response Completeness'),
    ])
    
    class Meta:
        db_table = 'rag_feedback'
        indexes = [
            models.Index(fields=['rating']),
            models.Index(fields=['feedback_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Feedback: {self.rating}/5 for {self.query}"