#!/usr/bin/env python3
"""
Simplified MCP Server Startup Script
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_server import main

if __name__ == '__main__':
    print("🚀 Starting Local MCP Server...")
    print("📝 Make sure you have set your API keys in the .env file:")
    print("   - OPENAI_API_KEY=your_key_here")
    print("   - ANTHROPIC_API_KEY=your_key_here")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)