from django.urls import path
from rag_core.views import VectorSearchView, DocumentUploadView

urlpatterns = [
    path('search/', VectorSearchView.as_view(), name='vector-search'),
    path('upload/', DocumentUploadView.as_view(), name='document-upload'),
]