"""
Health check utilities for the RAG platform
"""

import asyncio
from typing import Dict, Any
from django.conf import settings

from .llm_client import UnifiedLLMClient


async def check_mcp_server_health() -> Dict[str, Any]:
    """Check if the MCP server is healthy and responding."""
    try:
        async with UnifiedLLMClient() as client:
            health_data = await client.check_health()
            return {
                'mcp_server': {
                    'status': health_data.get('status', 'unknown'),
                    'url': settings.MCP_SERVER_URL,
                    'providers': health_data.get('providers', {}),
                    'error': health_data.get('error')
                }
            }
    except Exception as e:
        return {
            'mcp_server': {
                'status': 'error',
                'url': settings.MCP_SERVER_URL,
                'error': str(e)
            }
        }


async def check_llm_connectivity() -> Dict[str, Any]:
    """Test basic LLM connectivity through the MCP server."""
    try:
        async with UnifiedLLMClient() as client:
            # Simple test message
            test_messages = [
                {'role': 'user', 'content': 'Hello! This is a connectivity test.'}
            ]
            
            result = await client.chat_completion(
                messages=test_messages,
                max_tokens=50,
                temperature=0
            )
            
            if 'error' in result:
                return {
                    'llm_connectivity': {
                        'status': 'error',
                        'error': result['error']['message']
                    }
                }
            else:
                return {
                    'llm_connectivity': {
                        'status': 'healthy',
                        'test_response': result['choices'][0]['message']['content'][:100] + '...'
                    }
                }
    except Exception as e:
        return {
            'llm_connectivity': {
                'status': 'error',
                'error': str(e)
            }
        }


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Run health checks
        mcp_health = loop.run_until_complete(check_mcp_server_health())
        llm_health = loop.run_until_complete(check_llm_connectivity())
        
        # Combine results
        results = {**mcp_health, **llm_health}
        
        # Determine overall status
        overall_status = 'healthy'
        for check_name, check_data in results.items():
            if check_data.get('status') != 'healthy':
                overall_status = 'degraded'
                break
        
        results['overall_status'] = overall_status
        return results
        
    except Exception as e:
        return {
            'overall_status': 'error',
            'error': str(e),
            'mcp_server': {'status': 'error', 'error': str(e)},
            'llm_connectivity': {'status': 'error', 'error': str(e)}
        }