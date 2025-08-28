from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    RAGSessionViewSet,
    RAGQueryView,
    VectorSearchView,
    GraphRAGView,
    HybridSearchView,
    DocumentUploadView,
    RAGFeedbackView
)
from .health_views import HealthCheckView, ReadinessCheckView, LivenessCheckView, simple_health_check

router = DefaultRouter()
router.register(r'sessions', RAGSessionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('query/', RAGQueryView.as_view(), name='rag-query'),
    path('feedback/', RAGFeedbackView.as_view(), name='rag-feedback'),
    
    # Health check endpoints
    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('health/ready/', ReadinessCheckView.as_view(), name='readiness-check'),
    path('health/live/', LivenessCheckView.as_view(), name='liveness-check'),
    path('ping/', simple_health_check, name='simple-health'),
]