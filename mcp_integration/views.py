from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import MCPServer, MCPRequest, MCPTool
from .mcp_client import MCPManager, MCPClient


class MCPServerListView(APIView):
    """List and manage MCP servers."""
    
    def get(self, request):
        servers = MCPServer.objects.filter(is_active=True)
        data = [
            {
                'id': server.id,
                'name': server.name,
                'url': server.url,
                'server_type': server.server_type,
                'is_default': server.is_default,
                'capabilities': server.capabilities,
                'created_at': server.created_at
            }
            for server in servers
        ]
        return Response(data)
    
    def post(self, request):
        """Create a new MCP server configuration."""
        try:
            server = MCPServer.objects.create(
                name=request.data.get('name'),
                url=request.data.get('url'),
                api_key=request.data.get('api_key', ''),
                server_type=request.data.get('server_type', 'claude'),
                capabilities=request.data.get('capabilities', []),
                configuration=request.data.get('configuration', {}),
                is_default=request.data.get('is_default', False),
                max_context_length=request.data.get('max_context_length', 8192)
            )
            
            return Response({
                'id': server.id,
                'name': server.name,
                'message': 'MCP server created successfully'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {'error': f'Failed to create MCP server: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST
            )


class MCPRequestView(APIView):
    """Handle MCP requests."""
    
    def post(self, request):
        """Send a request to an MCP server."""
        try:
            import asyncio
            server_id = request.data.get('server_id')
            request_type = request.data.get('request_type', 'chat')
            prompt = request.data.get('prompt', '')
            parameters = request.data.get('parameters', {})
            
            # Get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if not server_id:
                server = loop.run_until_complete(MCPManager.get_default_server())
                if not server:
                    return Response(
                        {'error': 'No MCP server specified or configured'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            else:
                try:
                    server = MCPServer.objects.get(id=server_id, is_active=True)
                except MCPServer.DoesNotExist:
                    return Response(
                        {'error': 'MCP server not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            # Create and execute request
            mcp_request = loop.run_until_complete(MCPManager.create_request(
                server=server,
                request_type=request_type,
                prompt=prompt,
                parameters=parameters
            ))
            
            response_data = {
                'request_id': mcp_request.id,
                'status': mcp_request.status,
                'response': mcp_request.response,
                'processing_time': mcp_request.processing_time,
                'token_usage': mcp_request.token_usage,
                'error_message': mcp_request.error_message
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'MCP request failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MCPToolsView(APIView):
    """Manage and execute MCP tools."""
    
    def get(self, request):
        """List available MCP tools."""
        server_id = request.query_params.get('server_id')
        
        queryset = MCPTool.objects.filter(is_enabled=True, server__is_active=True)
        if server_id:
            queryset = queryset.filter(server_id=server_id)
        
        tools = queryset.select_related('server')
        data = [
            {
                'id': tool.id,
                'name': tool.name,
                'description': tool.description,
                'server_name': tool.server.name,
                'parameters_schema': tool.parameters_schema,
                'usage_count': tool.usage_count
            }
            for tool in tools
        ]
        
        return Response(data)
    
    def post(self, request):
        """Execute an MCP tool."""
        try:
            import asyncio
            tool_id = request.data.get('tool_id')
            tool_name = request.data.get('tool_name')
            parameters = request.data.get('parameters', {})
            server_id = request.data.get('server_id')
            
            # Get tool and server
            if tool_id:
                try:
                    tool = MCPTool.objects.get(id=tool_id, is_enabled=True)
                    server = tool.server
                except MCPTool.DoesNotExist:
                    return Response(
                        {'error': 'Tool not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            elif tool_name and server_id:
                try:
                    server = MCPServer.objects.get(id=server_id, is_active=True)
                    tool = MCPTool.objects.get(name=tool_name, server=server, is_enabled=True)
                except (MCPServer.DoesNotExist, MCPTool.DoesNotExist):
                    return Response(
                        {'error': 'Tool or server not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                return Response(
                    {'error': 'tool_id or (tool_name and server_id) required'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Execute tool
            async def execute_tool_async():
                async with MCPClient(server) as client:
                    return await client.execute_tool(tool.name, parameters)
            
            result = loop.run_until_complete(execute_tool_async())
            
            # Update usage count
            tool.usage_count = tool.usage_count + 1
            tool.save()
            
            response_data = {
                'tool_name': tool.name,
                'server_name': server.name,
                'parameters': parameters,
                'result': result,
                'success': 'error' not in result
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': f'Tool execution failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )