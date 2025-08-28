#!/usr/bin/env python3
"""
Local MCP (Model Context Protocol) Server
Acts as a translation layer between MCP requests and various LLM provider APIs
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from aiohttp import web, web_response
import openai
from anthropic import AsyncAnthropic
import cohere

# Configuration
@dataclass
class ServerConfig:
    host: str = "localhost"
    port: int = 8080
    debug: bool = True
    
    # Provider API Keys (loaded from environment)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cohere_api_key: str = ""
    huggingface_api_key: str = ""


class MCPServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = web.Application()
        self.setup_routes()
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.cohere_client = None
        
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize LLM provider clients based on available API keys."""
        if self.config.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.config.openai_api_key
            )
        
        if self.config.anthropic_api_key:
            self.anthropic_client = AsyncAnthropic(
                api_key=self.config.anthropic_api_key
            )
        
        if self.config.cohere_api_key:
            self.cohere_client = cohere.AsyncClient(
                api_key=self.config.cohere_api_key
            )
    
    def setup_routes(self):
        """Setup MCP-compatible API routes."""
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
        
        # CORS middleware
        self.app.middlewares.append(self.cors_middleware)
    
    @web.middleware
    async def cors_middleware(self, request: web.Request, handler):
        """Handle CORS for cross-origin requests."""
        if request.method == 'OPTIONS':
            response = web_response.Response()
        else:
            response = await handler(request)
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        """Handle MCP chat completion requests."""
        try:
            data = await request.json()
            
            # Extract MCP parameters
            messages = data.get('messages', [])
            model = data.get('model', 'gpt-3.5-turbo')
            temperature = data.get('temperature', 0.7)
            max_tokens = data.get('max_tokens', 2000)
            stream = data.get('stream', False)
            tools = data.get('tools', [])
            
            # Route to appropriate provider based on model
            if stream:
                return await self._stream_chat_completion(
                    messages, model, temperature, max_tokens, tools
                )
            else:
                return await self._chat_completion(
                    messages, model, temperature, max_tokens, tools
                )
        
        except Exception as e:
            return web.json_response({
                'error': {
                    'message': str(e),
                    'type': 'invalid_request_error'
                }
            }, status=400)
    
    async def _chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: List[Dict]
    ) -> web.Response:
        """Handle non-streaming chat completion."""
        start_time = time.time()
        
        try:
            # Route based on model prefix
            if model.startswith(('gpt-', 'text-davinci', 'text-ada')):
                response = await self._openai_chat_completion(
                    messages, model, temperature, max_tokens, tools
                )
            elif model.startswith(('claude-', 'anthropic')):
                response = await self._anthropic_chat_completion(
                    messages, model, temperature, max_tokens
                )
            elif model.startswith('command'):
                response = await self._cohere_chat_completion(
                    messages, model, temperature, max_tokens
                )
            else:
                # Default to OpenAI format
                response = await self._openai_chat_completion(
                    messages, model, temperature, max_tokens, tools
                )
            
            processing_time = time.time() - start_time
            response['processing_time'] = processing_time
            
            return web.json_response(response)
            
        except Exception as e:
            return web.json_response({
                'error': {
                    'message': f'Chat completion failed: {str(e)}',
                    'type': 'api_error'
                }
            }, status=500)
    
    async def _openai_chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int,
        tools: List[Dict]
    ) -> Dict:
        """Handle OpenAI chat completion."""
        if not self.openai_client:
            raise Exception("OpenAI client not configured")
        
        completion_args = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if tools:
            completion_args['tools'] = tools
        
        response = await self.openai_client.chat.completions.create(**completion_args)
        
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
    
    async def _anthropic_chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int
    ) -> Dict:
        """Handle Anthropic chat completion."""
        if not self.anthropic_client:
            raise Exception("Anthropic client not configured")
        
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
        
        response = await self.anthropic_client.messages.create(
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
    
    async def _cohere_chat_completion(
        self, 
        messages: List[Dict], 
        model: str, 
        temperature: float, 
        max_tokens: int
    ) -> Dict:
        """Handle Cohere chat completion."""
        if not self.cohere_client:
            raise Exception("Cohere client not configured")
        
        # Convert messages to Cohere format
        chat_history = []
        user_message = ""
        
        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
            elif msg['role'] == 'assistant':
                chat_history.append({
                    'role': 'CHATBOT',
                    'message': msg['content']
                })
            elif msg['role'] == 'system':
                chat_history.append({
                    'role': 'SYSTEM',
                    'message': msg['content']
                })
        
        response = await self.cohere_client.chat(
            model=model.replace('command', 'command-r'),
            message=user_message,
            chat_history=chat_history,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            'id': response.generation_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response.text,
                        'tool_calls': None
                    },
                    'finish_reason': response.finish_reason
                }
            ],
            'usage': {
                'prompt_tokens': 0,  # Cohere doesn't provide token counts in the same format
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }
    
    async def _stream_chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: List[Dict]
    ) -> web.Response:
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
            if model.startswith(('gpt-', 'text-')):
                async for chunk in self._openai_stream(messages, model, temperature, max_tokens, tools):
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            elif model.startswith(('claude-', 'anthropic')):
                async for chunk in self._anthropic_stream(messages, model, temperature, max_tokens):
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
    
    async def _openai_stream(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: List[Dict]
    ) -> AsyncGenerator[Dict, None]:
        """Stream OpenAI responses."""
        if not self.openai_client:
            raise Exception("OpenAI client not configured")
        
        completion_args = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True
        }
        
        if tools:
            completion_args['tools'] = tools
        
        stream = await self.openai_client.chat.completions.create(**completion_args)
        
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
    
    async def _anthropic_stream(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[Dict, None]:
        """Stream Anthropic responses."""
        # Anthropic streaming implementation would go here
        # For now, fall back to non-streaming and chunk the response
        response = await self._anthropic_chat_completion(messages, model, temperature, max_tokens)
        
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
            await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    async def completions(self, request: web.Request) -> web.Response:
        """Handle legacy completion requests."""
        data = await request.json()
        
        # Convert to chat format
        messages = [{'role': 'user', 'content': data.get('prompt', '')}]
        
        return await self._chat_completion(
            messages,
            data.get('model', 'gpt-3.5-turbo'),
            data.get('temperature', 0.7),
            data.get('max_tokens', 2000),
            []
        )
    
    async def embeddings(self, request: web.Request) -> web.Response:
        """Handle embedding requests."""
        try:
            data = await request.json()
            input_texts = data.get('input', [])
            model = data.get('model', 'text-embedding-ada-002')
            
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            
            if model.startswith('text-embedding') and self.openai_client:
                response = await self.openai_client.embeddings.create(
                    model=model,
                    input=input_texts
                )
                
                return web.json_response({
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
                })
            
            else:
                return web.json_response({
                    'error': {
                        'message': f'Embedding model {model} not supported',
                        'type': 'invalid_request_error'
                    }
                }, status=400)
        
        except Exception as e:
            return web.json_response({
                'error': {
                    'message': str(e),
                    'type': 'api_error'
                }
            }, status=500)
    
    async def list_models(self, request: web.Request) -> web.Response:
        """List available models."""
        models = []
        
        if self.openai_client:
            models.extend([
                {'id': 'gpt-4', 'object': 'model', 'owned_by': 'openai'},
                {'id': 'gpt-3.5-turbo', 'object': 'model', 'owned_by': 'openai'},
                {'id': 'text-embedding-ada-002', 'object': 'model', 'owned_by': 'openai'},
            ])
        
        if self.anthropic_client:
            models.extend([
                {'id': 'claude-3-opus-20240229', 'object': 'model', 'owned_by': 'anthropic'},
                {'id': 'claude-3-sonnet-20240229', 'object': 'model', 'owned_by': 'anthropic'},
            ])
        
        if self.cohere_client:
            models.extend([
                {'id': 'command-r', 'object': 'model', 'owned_by': 'cohere'},
                {'id': 'command-r-plus', 'object': 'model', 'owned_by': 'cohere'},
            ])
        
        return web.json_response({
            'object': 'list',
            'data': models
        })
    
    async def list_tools(self, request: web.Request) -> web.Response:
        """List available MCP tools."""
        tools = [
            {
                'name': 'web_search',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Search query'}
                    },
                    'required': ['query']
                }
            },
            {
                'name': 'calculator',
                'description': 'Perform mathematical calculations',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'expression': {'type': 'string', 'description': 'Mathematical expression'}
                    },
                    'required': ['expression']
                }
            }
        ]
        
        return web.json_response({'tools': tools})
    
    async def execute_tool(self, request: web.Request) -> web.Response:
        """Execute a tool."""
        try:
            data = await request.json()
            tool_name = data.get('tool')
            parameters = data.get('parameters', {})
            
            if tool_name == 'calculator':
                try:
                    expression = parameters.get('expression', '')
                    # Safe evaluation (be careful in production)
                    result = eval(expression, {"__builtins__": {}}, {})
                    return web.json_response({
                        'tool': tool_name,
                        'result': str(result),
                        'success': True
                    })
                except Exception as e:
                    return web.json_response({
                        'tool': tool_name,
                        'error': str(e),
                        'success': False
                    })
            
            elif tool_name == 'web_search':
                # Placeholder implementation
                query = parameters.get('query', '')
                return web.json_response({
                    'tool': tool_name,
                    'result': f'Search results for: {query} (placeholder implementation)',
                    'success': True
                })
            
            else:
                return web.json_response({
                    'error': f'Unknown tool: {tool_name}'
                }, status=400)
        
        except Exception as e:
            return web.json_response({
                'error': str(e)
            }, status=500)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'providers': {
                'openai': self.openai_client is not None,
                'anthropic': self.anthropic_client is not None,
                'cohere': self.cohere_client is not None
            }
        })
    
    async def server_info(self, request: web.Request) -> web.Response:
        """Server information endpoint."""
        return web.json_response({
            'name': 'Local MCP Server',
            'version': '1.0.0',
            'capabilities': [
                'chat.completions',
                'completions',
                'embeddings',
                'tools.execution',
                'streaming'
            ],
            'providers': {
                'openai': self.openai_client is not None,
                'anthropic': self.anthropic_client is not None,
                'cohere': self.cohere_client is not None
            }
        })


async def main():
    """Run the MCP server."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = ServerConfig(
        host=os.getenv('MCP_HOST', 'localhost'),
        port=int(os.getenv('MCP_PORT', '8080')),
        debug=os.getenv('DEBUG', 'True').lower() == 'true',
        openai_api_key=os.getenv('OPENAI_API_KEY', ''),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
        cohere_api_key=os.getenv('COHERE_API_KEY', ''),
        huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY', '')
    )
    
    server = MCPServer(config)
    
    print(f"🚀 Starting MCP Server on {config.host}:{config.port}")
    print(f"📋 Available providers:")
    print(f"   - OpenAI: {'✅' if config.openai_api_key else '❌'}")
    print(f"   - Anthropic: {'✅' if config.anthropic_api_key else '❌'}")
    print(f"   - Cohere: {'✅' if config.cohere_api_key else '❌'}")
    print(f"🔗 Health check: http://{config.host}:{config.port}/health")
    
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()
    
    print(f"✅ MCP Server running! Press Ctrl+C to stop.")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MCP Server...")
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())