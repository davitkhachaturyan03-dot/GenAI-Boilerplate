from django.urls import path
from . import views

urlpatterns = [
    # Query operations
    path('query/', views.GraphRAGQueryView.as_view(), name='graphrag-query'),
]
