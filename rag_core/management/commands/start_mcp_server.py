"""
Django management command to start the MCP server
"""

import asyncio
import os
import sys
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Start the local MCP server'

    def add_arguments(self, parser):
        parser.add_argument(
            '--host',
            type=str,
            default=getattr(settings, 'MCP_HOST', 'localhost'),
            help='Host to bind the server to'
        )
        parser.add_argument(
            '--port',
            type=int,
            default=getattr(settings, 'MCP_PORT', 8080),
            help='Port to bind the server to'
        )
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug mode'
        )

    def handle(self, *args, **options):
        # Add project root to path
        project_root = Path(settings.BASE_DIR)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            # Import and run the MCP server
            from mcp_server import ServerConfig, MCPServer
            
            config = ServerConfig(
                host=options['host'],
                port=options['port'],
                debug=options['debug'] or settings.DEBUG,
                openai_api_key=os.getenv('OPENAI_API_KEY', ''),
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY', ''),
                cohere_api_key=os.getenv('COHERE_API_KEY', ''),
                huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY', '')
            )
            
            self.stdout.write(
                self.style.SUCCESS(f'🚀 Starting MCP Server on {config.host}:{config.port}')
            )
            
            # Run the server
            asyncio.run(self._run_server(config))
            
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\n👋 MCP Server stopped by user'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed to start MCP Server: {e}')
            )
            sys.exit(1)

    async def _run_server(self, config):
        """Run the MCP server async."""
        from aiohttp import web
        from mcp_server import MCPServer
        
        server = MCPServer(config)
        
        self.stdout.write(f"📋 Available providers:")
        self.stdout.write(f"   - OpenAI: {'✅' if config.openai_api_key else '❌'}")
        self.stdout.write(f"   - Anthropic: {'✅' if config.anthropic_api_key else '❌'}")
        self.stdout.write(f"   - Cohere: {'✅' if config.cohere_api_key else '❌'}")
        self.stdout.write(f"🔗 Health check: http://{config.host}:{config.port}/health")
        
        runner = web.AppRunner(server.app)
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()
        
        self.stdout.write(self.style.SUCCESS('✅ MCP Server running! Press Ctrl+C to stop.'))
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            pass
        finally:
            await runner.cleanup()