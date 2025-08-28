#!/usr/bin/env python
"""
Script to create initial migrations for the RAG platform.
Run this after setting up the database and before starting the server.
"""

import os
import sys
import django
from django.core.management import execute_from_command_line

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
    django.setup()
    
    # Create migrations for all apps
    apps = ['vector_store', 'graph_rag', 'rag_core', 'mcp_integration']
    
    print("Creating migrations for all apps...")
    
    for app in apps:
        print(f"Creating migration for {app}...")
        execute_from_command_line(['manage.py', 'makemigrations', app])
    
    print("\nAll migrations created successfully!")
    print("Next steps:")
    print("1. Review the migration files")
    print("2. Run: python manage.py migrate")
    print("3. Create a superuser: python manage.py createsuperuser")
    print("4. Start the server: python manage.py runserver")