from django.urls import path
from .views import MCPServerListView, MCPRequestView, MCPToolsView

urlpatterns = [
    path('servers/', MCPServerListView.as_view(), name='mcp-servers'),
    path('request/', MCPRequestView.as_view(), name='mcp-request'),
    path('tools/', MCPToolsView.as_view(), name='mcp-tools'),
]