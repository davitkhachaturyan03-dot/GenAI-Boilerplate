from django.db import models
from vector_store.models import BaseModel


class MCPServer(BaseModel):
    name = models.CharField(max_length=100, unique=True)
    url = models.URLField()
    api_key = models.CharField(max_length=255, blank=True)
    server_type = models.CharField(max_length=50, default='claude', choices=[
        ('claude', 'Claude MCP'),
        ('openai', 'OpenAI Compatible'),
        ('custom', 'Custom MCP'),
    ])
    capabilities = models.JSONField(default=list, blank=True)
    configuration = models.JSONField(default=dict, blank=True)
    is_default = models.BooleanField(default=False)
    max_context_length = models.IntegerField(default=8192)
    
    class Meta:
        db_table = 'mcp_servers'

    def __str__(self):
        return self.name


class MCPRequest(BaseModel):
    server = models.ForeignKey(MCPServer, on_delete=models.CASCADE, related_name='requests')
    request_type = models.CharField(max_length=50, choices=[
        ('completion', 'Text Completion'),
        ('chat', 'Chat Completion'),
        ('embedding', 'Embedding'),
        ('function_call', 'Function Call'),
    ])
    prompt = models.TextField()
    parameters = models.JSONField(default=dict, blank=True)
    response = models.TextField(blank=True)
    response_metadata = models.JSONField(default=dict, blank=True)
    token_usage = models.JSONField(default=dict, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    error_message = models.TextField(blank=True)
    
    class Meta:
        db_table = 'mcp_requests'
        indexes = [
            models.Index(fields=['server', 'status']),
            models.Index(fields=['request_type', 'created_at']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"{self.request_type} request to {self.server.name}"


class MCPTool(BaseModel):
    name = models.CharField(max_length=100)
    server = models.ForeignKey(MCPServer, on_delete=models.CASCADE, related_name='tools')
    description = models.TextField(blank=True)
    parameters_schema = models.JSONField(default=dict, blank=True)
    is_enabled = models.BooleanField(default=True)
    usage_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'mcp_tools'
        unique_together = ['name', 'server']

    def __str__(self):
        return f"{self.name} ({self.server.name})"


class MCPToolExecution(BaseModel):
    tool = models.ForeignKey(MCPTool, on_delete=models.CASCADE, related_name='executions')
    request = models.ForeignKey(MCPRequest, on_delete=models.CASCADE, related_name='tool_executions')
    parameters = models.JSONField(default=dict, blank=True)
    result = models.TextField(blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('executing', 'Executing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    error_message = models.TextField(blank=True)
    
    class Meta:
        db_table = 'mcp_tool_executions'
        indexes = [
            models.Index(fields=['tool', 'status']),
            models.Index(fields=['status', 'created_at']),
        ]

    def __str__(self):
        return f"{self.tool.name} execution"