#!/usr/bin/env python3
"""
Test script for the local MCP server
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any


async def test_mcp_server(base_url: str = "http://localhost:8080"):
    """Test the MCP server endpoints."""
    
    print(f"🧪 Testing MCP Server at {base_url}")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health Check
        print("1️⃣ Testing Health Check...")
        try:
            async with session.get(f"{base_url}/health") as response:
                health_data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Health: {health_data.get('status')}")
                print(f"   Providers: {health_data.get('providers')}")
                print("   ✅ Health check passed\n")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}\n")
            return
        
        # Test 2: Server Info
        print("2️⃣ Testing Server Info...")
        try:
            async with session.get(f"{base_url}/info") as response:
                info_data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Name: {info_data.get('name')}")
                print(f"   Version: {info_data.get('version')}")
                print(f"   Capabilities: {info_data.get('capabilities')}")
                print("   ✅ Server info retrieved\n")
        except Exception as e:
            print(f"   ❌ Server info failed: {e}\n")
        
        # Test 3: List Models
        print("3️⃣ Testing List Models...")
        try:
            async with session.get(f"{base_url}/v1/models") as response:
                models_data = await response.json()
                print(f"   Status: {response.status}")
                models = models_data.get('data', [])
                print(f"   Available models: {len(models)}")
                for model in models[:3]:  # Show first 3
                    print(f"     - {model.get('id')} ({model.get('owned_by')})")
                print("   ✅ Models listed\n")
        except Exception as e:
            print(f"   ❌ List models failed: {e}\n")
        
        # Test 4: List Tools
        print("4️⃣ Testing List Tools...")
        try:
            async with session.get(f"{base_url}/v1/tools") as response:
                tools_data = await response.json()
                print(f"   Status: {response.status}")
                tools = tools_data.get('tools', [])
                print(f"   Available tools: {len(tools)}")
                for tool in tools:
                    print(f"     - {tool.get('name')}: {tool.get('description')}")
                print("   ✅ Tools listed\n")
        except Exception as e:
            print(f"   ❌ List tools failed: {e}\n")
        
        # Test 5: Execute Tool (Calculator)
        print("5️⃣ Testing Tool Execution (Calculator)...")
        try:
            payload = {
                "tool": "calculator",
                "parameters": {
                    "expression": "2 + 2 * 3"
                }
            }
            
            async with session.post(
                f"{base_url}/v1/tools/execute",
                json=payload
            ) as response:
                result_data = await response.json()
                print(f"   Status: {response.status}")
                print(f"   Tool: {result_data.get('tool')}")
                print(f"   Result: {result_data.get('result')}")
                print(f"   Success: {result_data.get('success')}")
                print("   ✅ Tool execution test passed\n")
        except Exception as e:
            print(f"   ❌ Tool execution failed: {e}\n")
        
        # Test 6: Chat Completion (if any provider is available)
        print("6️⃣ Testing Chat Completion...")
        try:
            payload = {
                "model": "gpt-3.5-turbo",  # Will fallback if OpenAI not available
                "messages": [
                    {"role": "user", "content": "Hello! This is a test message."}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload
            ) as response:
                completion_data = await response.json()
                print(f"   Status: {response.status}")
                
                if 'error' in completion_data:
                    print(f"   Error: {completion_data['error']['message']}")
                    print("   ⚠️  Chat completion failed (likely no API keys configured)")
                else:
                    choices = completion_data.get('choices', [])
                    if choices:
                        content = choices[0]['message']['content']
                        print(f"   Response: {content[:100]}...")
                        print(f"   Model: {completion_data.get('model')}")
                        print(f"   Usage: {completion_data.get('usage')}")
                        print("   ✅ Chat completion test passed")
                print()
        except Exception as e:
            print(f"   ❌ Chat completion failed: {e}\n")
        
        # Test 7: Streaming (if supported)
        print("7️⃣ Testing Streaming Chat...")
        try:
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                "stream": True,
                "max_tokens": 50
            }
            
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload
            ) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    print("   Streaming response:")
                    content_parts = []
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content']:
                                        content_parts.append(delta['content'])
                                        print(f"     {delta['content']}", end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                    
                    print("\n   ✅ Streaming test passed")
                else:
                    print("   ⚠️  Streaming not available")
                print()
        except Exception as e:
            print(f"   ❌ Streaming test failed: {e}\n")


async def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MCP Server')
    parser.add_argument('--url', default='http://localhost:8080', 
                       help='MCP Server URL (default: http://localhost:8080)')
    args = parser.parse_args()
    
    await test_mcp_server(args.url)
    
    print("🎉 MCP Server testing complete!")
    print("\n📋 Next steps:")
    print("   1. Configure API keys in .env file")
    print("   2. Start the Django RAG application")
    print("   3. Set MCP_SERVER_URL=http://localhost:8080 in .env")
    print("   4. Test the full RAG pipeline")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)