#!/usr/bin/env python3
"""
Local MCP (Model Context Protocol) Server
Acts as a translation layer between MCP requests and various LLM provider APIs
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
from pathlib import Path

from aiohttp import web, web_response
import openai
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = True
    log_level: str = "INFO"
    
    # Provider API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Security settings
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Enhanced CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: [
        "Content-Type", 
        "Authorization", 
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ])
    cors_expose_headers: List[str] = field(default_factory=lambda: ["Content-Length", "X-Processing-Time"])
    cors_allow_credentials: bool = True
    cors_max_age: int = 86400  # 24 hours
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('cors_origins', pre=True)
    def validate_cors_origins(cls, v):
        """Parse CORS origins from environment or use default."""
        if isinstance(v, str):
            # If it's a string, it's likely from environment variable
            if v == '*':
                return ['*']
            else:
                return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["http://localhost:3000", "http://localhost:8080"]
    
    @validator('cors_methods', pre=True)
    def validate_cors_methods(cls, v):
        """Parse CORS methods from environment or use default."""
        if isinstance(v, str):
            return [method.strip().upper() for method in v.split(',') if method.strip()]
        elif isinstance(v, list):
            return [method.upper() for method in v]
        else:
            return ["GET", "POST", "OPTIONS", "PUT", "DELETE"]
    
    @validator('cors_headers', pre=True)
    def validate_cors_headers(cls, v):
        """Parse CORS headers from environment or use default."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(',') if header.strip()]
        elif isinstance(v, list):
            return v
        else:
            return [
                "Content-Type", 
                "Authorization", 
                "X-Requested-With",
                "Accept",
                "Origin",
                "Access-Control-Request-Method",
                "Access-Control-Request-Headers"
            ]


# Pydantic Models for Request/Response
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=4000)
    stream: bool = False
    tools: Optional[List[Dict]] = None
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v


class EmbeddingRequest(BaseModel):
    """Embedding request model."""
    input: Union[str, List[str]]
    model: str = "text-embedding-ada-002"


class ToolExecutionRequest(BaseModel):
    """Tool execution request model."""
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


# Error handling
class MCPError(Exception):
    """Base MCP error."""
    def __init__(self, message: str, error_type: str = "mcp_error", status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(message)


class ProviderError(MCPError):
    """Provider-specific error."""
    pass


class ValidationError(MCPError):
    """Validation error."""
    def __init__(self, message: str):
        super().__init__(message, "validation_error", 400)


# Provider abstraction
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Perform chat completion."""
        pass
    
    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream chat completion."""
        pass
    
    @abstractmethod
    async def create_embeddings(self, input_texts: List[str], model: str) -> Dict:
        """Create embeddings."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[Dict]:
        """Get list of supported models."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Handle OpenAI chat completion."""
        completion_args = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if tools:
            completion_args['tools'] = tools
        
        response = await self.client.chat.completions.create(**completion_args)
        
        return {
            'id': response.id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': response.model,
            'choices': [
                {
                    'index': choice.index,
                    'message': {
                        'role': choice.message.role,
                        'content': choice.message.content,
                        'tool_calls': choice.message.tool_calls
                    },
                    'finish_reason': choice.finish_reason
                }
                for choice in response.choices
            ],
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
    
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream OpenAI responses."""
        completion_args = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True
        }
        
        if tools:
            completion_args['tools'] = tools
        
        stream = await self.client.chat.completions.create(**completion_args)
        
        async for chunk in stream:
            if chunk.choices:
                yield {
                    'id': chunk.id,
                    'object': 'chat.completion.chunk',
                    'created': chunk.created,
                    'model': chunk.model,
                    'choices': [
                        {
                            'index': choice.index,
                            'delta': {
                                'role': choice.delta.role,
                                'content': choice.delta.content
                            },
                            'finish_reason': choice.finish_reason
                        }
                        for choice in chunk.choices
                    ]
                }
    
    async def create_embeddings(self, input_texts: List[str], model: str) -> Dict:
        """Create OpenAI embeddings."""
        response = await self.client.embeddings.create(
            model=model,
            input=input_texts
        )
        
        return {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'index': i,
                    'embedding': emb.embedding
                }
                for i, emb in enumerate(response.data)
            ],
            'model': model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
    
    def get_supported_models(self) -> List[Dict]:
        """Get OpenAI supported models."""
        return [
            {'id': 'gpt-4', 'object': 'model', 'owned_by': 'openai'},
            {'id': 'gpt-3.5-turbo', 'object': 'model', 'owned_by': 'openai'},
            {'id': 'text-embedding-ada-002', 'object': 'model', 'owned_by': 'openai'},
        ]


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """Handle Anthropic chat completion."""
        # Convert messages to Anthropic format
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                user_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        response = await self.client.messages.create(
            model=model.replace('anthropic/', ''),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=user_messages
        )
        
        return {
            'id': response.id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response.content[0].text if response.content else "",
                        'tool_calls': None
                    },
                    'finish_reason': response.stop_reason
                }
            ],
            'usage': {
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    async def stream_chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[Dict, None]:
        """Stream Anthropic responses."""
        # TODO: Implement proper Anthropic streaming
        # For now, fall back to non-streaming and chunk the response
        response = await self.chat_completion(messages, model, temperature, max_tokens, tools)
        
        content = response['choices'][0]['message']['content']
        words = content.split()
        
        for i, word in enumerate(words):
            yield {
                'id': response['id'],
                'object': 'chat.completion.chunk',
                'created': response['created'],
                'model': model,
                'choices': [
                    {
                        'index': 0,
                        'delta': {
                            'content': word + (' ' if i < len(words) - 1 else '')
                        },
                        'finish_reason': 'stop' if i == len(words) - 1 else None
                    }
                ]
            }
            await asyncio.sleep(0.05)
    
    async def create_embeddings(self, input_texts: List[str], model: str) -> Dict:
        """Create Anthropic embeddings."""
        # Anthropic doesn't have a direct embeddings API like OpenAI
        raise ProviderError("Embeddings not supported for Anthropic", "not_supported", 400)
    
    def get_supported_models(self) -> List[Dict]:
        """Get Anthropic supported models."""
        return [
            {'id': 'claude-3-opus-20240229', 'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-3-sonnet-20240229', 'object': 'model', 'owned_by': 'anthropic'},
        ]

# Tool system
class Tool(ABC):
    """Abstract base class for tools."""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """Tool parameters schema."""
        pass


class CalculatorTool(Tool):
    """Calculator tool implementation."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'expression': {'type': 'string', 'description': 'Mathematical expression'}
            },
            'required': ['expression']
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute calculator tool safely."""
        import ast
        import operator
        
        expression = parameters.get('expression', '')
        
        # Safe evaluation using ast.literal_eval for basic operations
        # This is much safer than eval()
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            
            # Define allowed operations
            allowed_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_node(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return allowed_operators[type(node.op)](eval_node(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {type(node).__name__}")
            
            result = eval_node(tree.body)
            return {'result': str(result), 'success': True}
            
        except Exception as e:
            return {'error': str(e), 'success': False}


class WebSearchTool(Tool):
    """Web search tool implementation."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information"
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Search query'}
            },
            'required': ['query']
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search tool."""
        # TODO: Implement actual web search
        query = parameters.get('query', '')
        return {
            'result': f'Search results for: {query} (placeholder implementation)',
            'success': True
        }


# Provider manager
class ProviderManager:
    """Manages LLM providers."""
    
    def __init__(self, config: ServerConfig):
        self.providers: Dict[ProviderType, LLMProvider] = {}
        self._setup_providers(config)
    
    def _setup_providers(self, config: ServerConfig):
        """Setup available providers."""

        if config.openai_api_key:
            self.providers[ProviderType.OPENAI] = OpenAIProvider(config.openai_api_key)
        
        if config.anthropic_api_key:
            self.providers[ProviderType.ANTHROPIC] = AnthropicProvider(config.anthropic_api_key)
    
    def get_provider_for_model(self, model: str) -> Optional[LLMProvider]:
        """Get the appropriate provider for a given model."""

        if model.startswith(('gpt-', 'text-')):
            return self.providers.get(ProviderType.OPENAI)
        elif model.startswith(('claude-', 'anthropic')):
            return self.providers.get(ProviderType.ANTHROPIC)
        else:
            # Default to OpenAI
            return self.providers.get(ProviderType.OPENAI)
    
    def get_all_models(self) -> List[Dict]:
        """Get all supported models from all providers."""

        models = []

        for provider in self.providers.values():
            models.extend(provider.get_supported_models())

        return models


# Tool manager
class ToolManager:
    """Manages available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {
            'calculator': CalculatorTool(),
            'web_search': WebSearchTool(),
        }
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Dict]:
        """Get all available tools."""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters_schema
            }
            for tool in self.tools.values()
        ]


# Rate limiting
class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False


# Main server class
class MCPServer:
    """Main MCP server implementation."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = web.Application()
        self.provider_manager = ProviderManager(config)
        self.tool_manager = ToolManager()
        self.rate_limiter = RateLimiter(config.rate_limit_requests, config.rate_limit_window)
        
        self.setup_routes()
        self.setup_middleware()
    
    def setup_routes(self):
        """Setup API routes."""
        # MCP Standard Endpoints
        self.app.router.add_post('/v1/chat/completions', self.chat_completions)
        self.app.router.add_post('/v1/completions', self.completions)
        self.app.router.add_post('/v1/embeddings', self.embeddings)
        self.app.router.add_get('/v1/models', self.list_models)
        
        # MCP Tool Execution
        self.app.router.add_get('/v1/tools', self.list_tools)
        self.app.router.add_post('/v1/tools/execute', self.execute_tool)
        
        # Health and Info
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/info', self.server_info)
    
    def setup_middleware(self):
        """Setup middleware."""
        self.app.middlewares.extend([
            self.timing_middleware,
            self.error_middleware,
            self.rate_limit_middleware,
            self.cors_middleware,
        ])
    
    @web.middleware
    async def timing_middleware(self, request: web.Request, handler):
        """Add timing information to requests."""
        request.start_time = time.time()
        return await handler(request)

    @web.middleware
    async def cors_middleware(self, request: web.Request, handler):
        """Enhanced CORS middleware with better security and flexibility."""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = web_response.Response()
            response.headers['Access-Control-Max-Age'] = str(self.config.cors_max_age)
        else:
            response = await handler(request)

        # Get the origin from the request
        origin = request.headers.get('Origin', '')

        # Determine the appropriate CORS origin
        if '*' in self.config.cors_origins:
            # Allow all origins if '*' is in the list (not recommended for production)
            cors_origin = '*'
            # When using wildcard, credentials must be false
            allow_credentials = 'false'
        elif origin in self.config.cors_origins:
            # Allow specific origin if it's in the allowed list
            cors_origin = origin
            allow_credentials = 'true' if self.config.cors_allow_credentials else 'false'
        else:
            # Origin not in allowed list - deny access
            cors_origin = ''
            allow_credentials = 'false'
        
        # Set CORS headers
        if cors_origin:
            response.headers['Access-Control-Allow-Origin'] = cors_origin
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.config.cors_methods)
            response.headers['Access-Control-Allow-Headers'] = ', '.join(self.config.cors_headers)
            response.headers['Access-Control-Allow-Credentials'] = allow_credentials
            
            # Set exposed headers
            if self.config.cors_expose_headers:
                response.headers['Access-Control-Expose-Headers'] = ', '.join(self.config.cors_expose_headers)
        else:
            # Log denied origin for debugging
            logger.debug(f"CORS: Origin '{origin}' not in allowed list: {self.config.cors_origins}")
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Add processing time header for performance monitoring
        if hasattr(request, 'start_time'):
            processing_time = time.time() - request.start_time
            response.headers['X-Processing-Time'] = f'{processing_time:.3f}'
        
        return response

    @web.middleware
    async def error_middleware(self, request: web.Request, handler):
        """Handle errors globally."""
        try:
            return await handler(request)
        except MCPError as e:
            return web.json_response({
                'error': {
                    'message': e.message,
                    'type': e.error_type
                }
            }, status=e.status_code)
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return web.json_response({
                'error': {
                    'message': 'Internal server error',
                    'type': 'internal_error'
                }
            }, status=500)

    @web.middleware
    async def rate_limit_middleware(self, request: web.Request, handler):
        """Handle rate limiting."""
        if self.config.enable_rate_limiting and not self.rate_limiter.is_allowed():
            return web.json_response({
                'error': {
                    'message': 'Rate limit exceeded',
                    'type': 'rate_limit_error'
                }
            }, status=429)
        
        return await handler(request)
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        """Handle chat completion requests."""
        data = await request.json()
        
        # Validate request
        try:
            chat_request = ChatCompletionRequest(**data)
        except Exception as e:
            raise ValidationError(f"Invalid request: {e}")
        
        # Get provider
        provider = self.provider_manager.get_provider_for_model(chat_request.model)
        if not provider:
            raise ProviderError(f"No provider available for model: {chat_request.model}")
        
        start_time = time.time()
        
        try:
            if chat_request.stream:
                return await self._stream_chat_completion(provider, chat_request)
            else:
                response = await provider.chat_completion(
                    [msg.dict() for msg in chat_request.messages],
                    chat_request.model,
                    chat_request.temperature,
                    chat_request.max_tokens,
                    chat_request.tools
                )
                response['processing_time'] = time.time() - start_time
                return web.json_response(response)
        
        except Exception as e:
            logger.error(f"Chat completion failed: {e}", exc_info=True)
            raise ProviderError(f"Chat completion failed: {str(e)}")
    
    async def _stream_chat_completion(self, provider: LLMProvider, request: ChatCompletionRequest) -> web.Response:
        """Handle streaming chat completion."""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/plain',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        
        await response.prepare(request)
        
        try:
            async for chunk in provider.stream_chat_completion(
                [msg.dict() for msg in request.messages],
                request.model,
                request.temperature,
                request.max_tokens,
                request.tools
            ):
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            
            await response.write(b"data: [DONE]\n\n")
            
        except Exception as e:
            error_chunk = {
                'error': {
                    'message': str(e),
                    'type': 'stream_error'
                }
            }
            await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
        
        return response
    
    async def completions(self, request: web.Request) -> web.Response:
        """Handle legacy completion requests."""
        try:
            data = await request.json()
            logger.debug(f"Completions request data: {data}")
            
            # Convert to chat format
            messages = [ChatMessage(role='user', content=data.get('prompt', ''))]
            
            chat_request = ChatCompletionRequest(
                messages=messages,
                model=data.get('model', 'gpt-3.5-turbo'),
                temperature=data.get('temperature', 0.7),
                max_tokens=data.get('max_tokens', 2000)
            )
            logger.debug(f"Created chat request: {chat_request}")
        except Exception as e:
            logger.error(f"Error parsing completions request: {e}", exc_info=True)
            raise ValidationError(f"Invalid request format: {e}")
        
        # Get provider
        logger.debug(f"Getting provider for model: {chat_request.model}")
        provider = self.provider_manager.get_provider_for_model(chat_request.model)
        logger.debug(f"Selected provider: {type(provider).__name__ if provider else 'None'}")
        
        if not provider:
            available_providers = list(self.provider_manager.providers.keys())
            if not available_providers:
                raise ProviderError("No LLM providers configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
            else:
                raise ProviderError(f"No provider available for model: {chat_request.model}. Available providers: {[p.value for p in available_providers]}")
        
        start_time = time.time()
        
        try:
            if chat_request.stream:
                return await self._stream_chat_completion(provider, chat_request)
            else:
                response = await provider.chat_completion(
                    [msg.dict() for msg in chat_request.messages],
                    chat_request.model,
                    chat_request.temperature,
                    chat_request.max_tokens,
                    chat_request.tools
                )
                response['processing_time'] = time.time() - start_time
                return web.json_response(response)
        
        except Exception as e:
            logger.error(f"Completion failed: {e}", exc_info=True)
            raise ProviderError(f"Completion failed: {str(e)}")
    
    async def embeddings(self, request: web.Request) -> web.Response:
        """Handle embedding requests."""
        data = await request.json()
        
        # Validate request
        try:
            embedding_request = EmbeddingRequest(**data)
        except Exception as e:
            raise ValidationError(f"Invalid request: {e}")
        
        # Get provider
        provider = self.provider_manager.get_provider_for_model(embedding_request.model)
        if not provider:
            available_providers = list(self.provider_manager.providers.keys())
            if not available_providers:
                raise ProviderError("No LLM providers configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
            else:
                raise ProviderError(f"No provider available for model: {embedding_request.model}. Available providers: {[p.value for p in available_providers]}")
        
        try:
            input_texts = [embedding_request.input] if isinstance(embedding_request.input, str) else embedding_request.input
            response = await provider.create_embeddings(input_texts, embedding_request.model)
            return web.json_response(response)
        
        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            raise ProviderError(f"Embedding failed: {str(e)}")
    
    async def list_models(self, request: web.Request) -> web.Response:
        """List available models."""
        models = self.provider_manager.get_all_models()
        return web.json_response({
            'object': 'list',
            'data': models
        })
    
    async def list_tools(self, request: web.Request) -> web.Response:
        """List available tools."""
        tools = self.tool_manager.get_all_tools()
        return web.json_response({'tools': tools})
    
    async def execute_tool(self, request: web.Request) -> web.Response:
        """Execute a tool."""
        data = await request.json()
        
        # Validate request
        try:
            tool_request = ToolExecutionRequest(**data)
        except Exception as e:
            raise ValidationError(f"Invalid request: {e}")
        
        # Get tool
        tool = self.tool_manager.get_tool(tool_request.tool)
        if not tool:
            raise ValidationError(f"Unknown tool: {tool_request.tool}")
        
        try:
            result = await tool.execute(tool_request.parameters)
            return web.json_response({
                'tool': tool_request.tool,
                **result
            })
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return web.json_response({
                'tool': tool_request.tool,
                'error': str(e),
                'success': False
            })
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'providers': {
                provider_type.value: provider is not None
                for provider_type, provider in self.provider_manager.providers.items()
            }
        })
    
    async def server_info(self, request: web.Request) -> web.Response:
        """Server information endpoint."""
        return web.json_response({
            'name': 'Local MCP Server',
            'version': '2.0.0',
            'capabilities': [
                'chat.completions',
                'completions',
                'embeddings',
                'tools.execution',
                'streaming'
            ],
            'providers': {
                provider_type.value: provider is not None
                for provider_type, provider in self.provider_manager.providers.items()
            }
        })


async def main():
    """Run the MCP server."""
    load_dotenv()
    
    # Parse CORS configuration from environment
    cors_origins_str = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080')
    cors_methods_str = os.getenv('CORS_METHODS', 'GET,POST,OPTIONS,PUT,DELETE')
    cors_headers_str = os.getenv('CORS_HEADERS', 'Content-Type,Authorization,X-Requested-With,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers')
    cors_expose_headers_str = os.getenv('CORS_EXPOSE_HEADERS', 'Content-Length,X-Processing-Time')
    
    config = ServerConfig(
        host=os.getenv('MCP_HOST', 'localhost'),
        port=int(os.getenv('MCP_PORT', '8080')),
        debug=os.getenv('DEBUG', 'True').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
        enable_rate_limiting=os.getenv('ENABLE_RATE_LIMITING', 'True').lower() == 'true',
        rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '100')),
        rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
        cors_origins=cors_origins_str,
        cors_methods=cors_methods_str,
        cors_headers=cors_headers_str,
        cors_expose_headers=cors_expose_headers_str,
        cors_allow_credentials=os.getenv('CORS_ALLOW_CREDENTIALS', 'True').lower() == 'true',
        cors_max_age=int(os.getenv('CORS_MAX_AGE', '86400'))
    )
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    # Log CORS configuration for debugging
    logger.info(f"🔧 CORS Configuration:")
    logger.info(f"   Origins: {config.cors_origins}")
    logger.info(f"   Methods: {config.cors_methods}")
    logger.info(f"   Headers: {config.cors_headers}")
    logger.info(f"   Allow Credentials: {config.cors_allow_credentials}")
    
    server = MCPServer(config)
    
    logger.info(f"🚀 Starting MCP Server on {config.host}:{config.port}")
    logger.info(f"📋 Available providers:")

    for provider_type in server.provider_manager.providers:
        logger.info(f"   - {provider_type.value}: ✅")

    logger.info(f"🔗 Health check: http://{config.host}:{config.port}/health")
    
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()
    
    logger.info(f"✅ MCP Server running! Press Ctrl+C to stop.")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down MCP Server...")
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())