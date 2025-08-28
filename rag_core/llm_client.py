"""
Unified LLM Client that forwards all requests to the local MCP server (synchronous only)
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional
from django.conf import settings


class UnifiedLLMClient:
    """
    Unified LLM client that routes all requests through the local MCP server.
    This provides a consistent interface regardless of the underlying LLM provider.
    All methods are synchronous.
    """
    
    def __init__(self):
        self.mcp_url = getattr(settings, 'MCP_SERVER_URL', 'http://localhost:8080')
        self.default_model = getattr(settings, 'DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')
        self.default_temperature = getattr(settings, 'LLM_TEMPERATURE', 0.7)
        self.default_max_tokens = getattr(settings, 'LLM_MAX_TOKENS', 2000)
        self.timeout = 300  # 5 minutes timeout
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the MCP server.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to DEFAULT_LLM_MODEL)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: List of available tools for function calling
            
        Returns:
            Dictionary containing the response data
        """
        payload = {
            'messages': messages,
            'model': model or self.default_model,
            'temperature': temperature or self.default_temperature,
            'max_tokens': max_tokens or self.default_max_tokens,
            'stream': False
        }
        
        if tools:
            payload['tools'] = tools
        
        try:
            response = requests.post(
                f"{self.mcp_url}/v1/chat/completions",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return {
                    'error': {
                        'message': f'MCP server error: {response.text}',
                        'type': 'mcp_error',
                        'status_code': response.status_code
                    }
                }
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                'error': {
                    'message': f'Connection error to MCP server: {str(e)}',
                    'type': 'connection_error'
                }
            }
        except Exception as e:
            return {
                'error': {
                    'message': f'Unexpected error: {str(e)}',
                    'type': 'unknown_error'
                }
            }
    
    def create_embedding(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Create embeddings using the MCP server.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            Dictionary containing embedding data
        """
        payload = {
            'input': texts,
            'model': model
        }
        
        try:
            response = requests.post(
                f"{self.mcp_url}/v1/embeddings",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return {
                    'error': {
                        'message': f'Embedding error: {response.text}',
                        'type': 'embedding_error'
                    }
                }
            
            return response.json()
        
        except Exception as e:
            return {
                'error': {
                    'message': f'Embedding request failed: {str(e)}',
                    'type': 'request_error'
                }
            }
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool via the MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        payload = {
            'tool': tool_name,
            'parameters': parameters
        }
        
        try:
            response = requests.post(
                f"{self.mcp_url}/v1/tools/execute",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            
            return response.json()
        
        except Exception as e:
            return {
                'error': {
                    'message': f'Tool execution failed: {str(e)}',
                    'type': 'tool_error'
                }
            }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from the MCP server.
        
        Returns:
            List of available models
        """
        try:
            response = requests.get(
                f"{self.mcp_url}/v1/models",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
            else:
                return []
        
        except Exception:
            return []
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the MCP server.
        
        Returns:
            Health status information
        """
        try:
            response = requests.get(
                f"{self.mcp_url}/health",
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'status': 'unhealthy',
                    'error': f'HTTP {response.status_code}'
                }
        
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e)
            }
    
    # Alias methods for backward compatibility
    def sync_chat_completion(self, *args, **kwargs):
        """Synchronous chat completion (same as chat_completion now)."""
        return self.chat_completion(*args, **kwargs)
    
    def sync_create_embedding(self, *args, **kwargs):
        """Synchronous embedding creation (same as create_embedding now)."""
        return self.create_embedding(*args, **kwargs)


# Global instance for easy importing
llm_client = UnifiedLLMClient()


# Convenience functions (now synchronous)
def generate_response(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Convenience function to generate a text response.
    
    Returns:
        Generated text or error message
    """
    client = UnifiedLLMClient()
    result = client.chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    if 'error' in result:
        return f"Error: {result['error']['message']}"
    
    choices = result.get('choices', [])
    if choices:
        return choices[0]['message']['content']
    
    return "No response generated"


def generate_rag_response(
    query: str,
    context: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    Convenience function for RAG responses.
    
    Args:
        query: User query
        context: Retrieved context
        system_prompt: Optional system prompt
        model: Model to use
        
    Returns:
        Generated response text
    """
    messages = []
    
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    
    rag_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context information above. If the context doesn't contain sufficient information to answer the question, please indicate what information is missing."""
    
    messages.append({'role': 'user', 'content': rag_prompt})
    
    return generate_response(messages, model=model)