import asyncio
import logging
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .graphrag_engine import MSGraphRAGEngine

logger = logging.getLogger(__name__)

class GraphRAGQueryView(APIView):
    """
    API view for querying GraphRAG projects.
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Execute a query against a GraphRAG project."""
        project_name = request.data.get('project_name', 'default')
        query_text = request.data.get('query')
        query_type = request.data.get('query_type', 'local')

        if not query_text:
            return Response(
                {'error': 'Query text is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Start query asynchronously
        asyncio.create_task(self._run_query(project_name, query_text, query_type))
        
        return Response({
            'message': 'Query started',
            'project_name': project_name,
            'query_text': query_text,
            'query_type': query_type
        }, status=status.HTTP_202_ACCEPTED)

    async def _run_query(self, project_name: str, query_text: str, query_type: str):
        """Run query in background."""
        try:
            engine = MSGraphRAGEngine(project_name)
            result = await engine.query_graphrag(
                query_text, query_type
            )
            
            if result['success']:
                logger.info(f"Query completed successfully for project {project_name}")
            else:
                logger.error(f"Query failed for project {project_name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
