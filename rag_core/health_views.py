"""
Health check views for monitoring the RAG platform
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.conf import settings

from .health_checks import run_health_checks


class HealthCheckView(APIView):
    """
    Health check endpoint for the RAG platform.
    Returns the status of all critical services.
    """
    
    def get(self, request):
        """Return health status of all services."""
        try:
            health_data = run_health_checks()
            
            # Determine HTTP status based on health
            overall_status = health_data.get('overall_status', 'unknown')
            http_status = status.HTTP_200_OK if overall_status == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
            
            return Response(health_data, status=http_status)
            
        except Exception as e:
            return Response({
                'overall_status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReadinessCheckView(APIView):
    """
    Readiness check - indicates if the service is ready to handle requests.
    """
    
    def get(self, request):
        """Check if the service is ready."""
        try:
            health_data = run_health_checks()
            mcp_status = health_data.get('mcp_server', {}).get('status', 'unknown')
            
            if mcp_status == 'healthy':
                return Response({
                    'status': 'ready',
                    'mcp_server': mcp_status
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'status': 'not_ready',
                    'mcp_server': mcp_status,
                    'details': health_data
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
        except Exception as e:
            return Response({
                'status': 'error',
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LivenessCheckView(APIView):
    """
    Liveness check - indicates if the service is alive.
    Simple check that doesn't depend on external services.
    """
    
    def get(self, request):
        """Simple liveness check."""
        return Response({
            'status': 'alive',
            'service': 'RAG Platform',
            'version': '1.0.0'
        }, status=status.HTTP_200_OK)


def simple_health_check(request):
    """Simple health check for load balancers."""
    return JsonResponse({
        'status': 'ok',
        'service': 'rag-platform'
    })