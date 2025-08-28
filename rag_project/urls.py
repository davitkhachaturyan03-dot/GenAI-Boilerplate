from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/rag/', include('rag_core.urls')),
    path('api/graph/', include('graph_rag.urls')),
    path('api/mcp/', include('mcp_integration.urls')),
    path('api/vectors/', include('vector_store.urls')),
]