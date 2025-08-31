import json
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from django.conf import settings
from asgiref.sync import sync_to_async

from .models import MCPServer, MCPRequest, MCPTool, MCPToolExecution


@dataclass
class MCPResponse:
    content: str
    metadata: Dict[str, Any]
    token_usage: Dict[str, int]
    processing_time: float
    status: str
    error: Optional[str] = None


class MCPClient:
    def __init__(self, server: MCPServer):
        self.server = server
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.server.api_key}' if self.server.api_key else None
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> MCPResponse:
        """Send a chat completion request to the MCP server."""
        payload = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream
        }
        
        if tools:
            payload['tools'] = tools
        
        # Add server-specific configurations
        payload.update(self.server.configuration.get('completion_params', {}))
        
        try:
            async with self.session.post(
                f"{self.server.url}/v1/chat/completions",
                json=payload
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    return MCPResponse(
                        content="",
                        metadata={},
                        token_usage={},
                        processing_time=0.0,
                        status="failed",
                        error=response_data.get('error', 'Unknown error')
                    )
                
                return MCPResponse(
                    content=response_data['choices'][0]['message']['content'],
                    metadata={
                        'model': response_data.get('model', ''),
                        'finish_reason': response_data['choices'][0].get('finish_reason', ''),
                        'response_id': response_data.get('id', '')
                    },
                    token_usage=response_data.get('usage', {}),
                    processing_time=response_data.get('processing_time', 0.0),
                    status="completed"
                )
                
        except Exception as e:
            return MCPResponse(
                content="",
                metadata={},
                token_usage={},
                processing_time=0.0,
                status="failed",
                error=str(e)
            )
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion from MCP server."""
        payload = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True
        }
        
        payload.update(self.server.configuration.get('completion_params', {}))
        
        try:
            async with self.session.post(
                f"{self.server.url}/v1/chat/completions",
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def create_embedding(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create embeddings using MCP server."""
        payload = {
            'input': texts,
            'model': model or self.server.configuration.get('embedding_model', 'text-embedding-ada-002')
        }
        
        try:
            async with self.session.post(
                f"{self.server.url}/v1/embeddings",
                json=payload
            ) as response:
                return await response.json()
                
        except Exception as e:
            return {'error': str(e)}
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool via MCP server."""
        payload = {
            'tool': tool_name,
            'parameters': parameters
        }
        
        try:
            async with self.session.post(
                f"{self.server.url}/v1/tools/execute",
                json=payload
            ) as response:
                return await response.json()
                
        except Exception as e:
            return {'error': str(e)}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server."""
        try:
            async with self.session.get(
                f"{self.server.url}/v1/tools"
            ) as response:
                response_data = await response.json()
                return response_data.get('tools', [])
                
        except Exception as e:
            return []


class MCPManager:
    """Manager class for handling multiple MCP servers and requests."""

    @staticmethod
    async def create_request(
        server: MCPServer,
        request_type: str,
        prompt: str,
        parameters: Dict[str, Any]
    ) -> MCPRequest:
        """Create and execute an MCP request."""
        request = await sync_to_async(MCPRequest.objects.create)(
            server=server,
            request_type=request_type,
            prompt=prompt,
            parameters=parameters,
            status='pending'
        )

        try:
            await sync_to_async(setattr)(request, 'status', 'processing')
            await sync_to_async(request.save)()

            async with MCPClient(server) as client:
                match request_type:
                    case 'chat':
                        messages = parameters.get('messages', [{'role': 'user', 'content': prompt}])
                        response = await client.chat_completion(
                            messages=messages,
                            temperature=parameters.get('temperature', 0.7),
                            max_tokens=parameters.get('max_tokens', 2000),
                            tools=parameters.get('tools')
                        )

                        await sync_to_async(setattr)(request, 'response', response.content)
                        await sync_to_async(setattr)(request, 'response_metadata', response.metadata)
                        await sync_to_async(setattr)(request, 'token_usage', response.token_usage)
                        await sync_to_async(setattr)(request, 'processing_time', response.processing_time)
                        await sync_to_async(setattr)(request, 'status', response.status)
                        await sync_to_async(setattr)(request, 'error_message', response.error or "")

                    case 'embedding':
                        texts = parameters.get('texts', [prompt])
                        response = await client.create_embedding(texts)
                        
                        if 'error' in response:
                            await sync_to_async(setattr)(request, 'status', 'failed')
                            await sync_to_async(setattr)(request, 'error_message', response['error'])
                        else:
                            await sync_to_async(setattr)(request, 'response', json.dumps(response))
                            await sync_to_async(setattr)(request, 'status', 'completed')

                    case 'function_call':
                        tool_name = parameters.get('tool_name', '')
                        tool_params = parameters.get('tool_parameters', {})
                        response = await client.execute_tool(tool_name, tool_params)
                        
                        if 'error' in response:
                            await sync_to_async(setattr)(request, 'status', 'failed')
                            await sync_to_async(setattr)(request, 'error_message', response['error'])
                        else:
                            await sync_to_async(setattr)(request, 'response', json.dumps(response))
                            await sync_to_async(setattr)(request, 'status', 'completed')

        except Exception as e:
            await sync_to_async(setattr)(request, 'status', 'failed')
            await sync_to_async(setattr)(request, 'error_message', str(e))
        
        finally:
            await sync_to_async(request.save)()
        
        return request
    
    @staticmethod
    async def get_default_server() -> Optional[MCPServer]:
        """Get the default MCP server."""
        try:
            return await sync_to_async(MCPServer.objects.filter(is_default=True, is_active=True).first)()
        except MCPServer.DoesNotExist:
            return None
    
    @staticmethod
    async def sync_server_tools(server: MCPServer):
        """Synchronize tools available on the MCP server."""
        async with MCPClient(server) as client:
            tools_data = await client.list_tools()
            
            # Update or create tools
            for tool_data in tools_data:
                tool, created = await sync_to_async(MCPTool.objects.get_or_create)(
                    name=tool_data['name'],
                    server=server,
                    defaults={
                        'description': tool_data.get('description', ''),
                        'parameters_schema': tool_data.get('parameters', {}),
                        'is_enabled': True
                    }
                )
                
                if not created:
                    await sync_to_async(setattr)(tool, 'description', tool_data.get('description', tool.description))
                    await sync_to_async(setattr)(tool, 'parameters_schema', tool_data.get('parameters', tool.parameters_schema))
                    await sync_to_async(tool.save)()
    
    @staticmethod
    async def execute_rag_with_mcp(
        server: MCPServer,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> MCPRequest:
        """Execute a RAG query using MCP server."""
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        # Construct the RAG prompt
        rag_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context information above."""
        
        messages.append({'role': 'user', 'content': rag_prompt})
        
        return await MCPManager.create_request(
            server=server,
            request_type='chat',
            prompt=rag_prompt,
            parameters={
                'messages': messages,
                'temperature': temperature,
                'max_tokens': 2000
            }
        )