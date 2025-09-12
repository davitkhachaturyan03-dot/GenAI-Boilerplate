import os
import subprocess
import logging
from typing import Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)

class MSGraphRAGEngine:
    """
    GraphRAG Engine that handles subprocess integration with Microsoft's GraphRAG package.
    """

    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.app_dir = os.path.dirname(__file__)
        self.app_settings_path = os.path.join(self.app_dir, 'settings.yaml')
        self.app_prompts_dir = os.path.join(self.app_dir, 'prompts')
        self.working_dir = self._setup_working_directory()
        self.config_path = os.path.join(self.working_dir, 'settings.yaml')
 
    def _setup_working_directory(self) -> str:
        """Set up the working directory for GraphRAG operations."""
        base_dir = getattr(settings, 'GRAPH_RAG_BASE_DIR', self.app_dir)
        project_dir = os.path.join(base_dir, self.project_name)
        
        # Create directories
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'output'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'cache'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'prompts'), exist_ok=True)
        
        return project_dir

    async def query_graphrag(self, query_text: str, query_type: str = 'local') -> Dict[str, Any]:
        """
        Query GraphRAG using subprocess.
        """
        try:
            # Setup environment
            env = os.environ.copy()
            env['OPENAI_API_KEY'] = settings.OPENAI_API_KEY

            # Build query command; use app_dir as root since it contains settings/prompts/outputs
            cmd = [
                'graphrag',
                'query',
                '--root', self.app_dir,
                '--method', query_type,
                '--query', query_text
            ]

            # Log the command being run
            logger.info(f"Running GraphRAG query command: {' '.join(cmd)}")

            # Run query
            process = subprocess.Popen(
                cmd,
                cwd=self.app_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            stdout, stderr = process.communicate()
            return_code = process.returncode
            
            if return_code == 0:
                logger.info(f"GraphRAG query completed successfully for project {self.project_name}")
                return {
                    'success': True,
                    'response': stdout,
                    'query_type': query_type,
                    'query_text': query_text
                }
            else:
                error_msg = f"GraphRAG query failed with return code {return_code}: {stderr}"
                logger.error(f"GraphRAG query failed for project {self.project_name}: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'query_type': query_type,
                    'query_text': query_text
                }
            
        except Exception as e:
            error_msg = f"Error during GraphRAG query: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'query_type': query_type,
                'query_text': query_text
            }
