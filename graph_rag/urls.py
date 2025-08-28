from django.urls import path
from rag_core.views import GraphRAGView, HybridSearchView

urlpatterns = [
    path('query/', GraphRAGView.as_view(), name='graph-rag-query'),
    path('hybrid/', HybridSearchView.as_view(), name='hybrid-search'),
]