import os
import asyncio
import json
from pathlib import Path

import pandas as pd


def setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_project.settings")
    # Lazy import to avoid requiring Django when this module is imported elsewhere
    import django  # type: ignore

    django.setup()


async def run_query(project_name: str, query_text: str, query_type: str = "local") -> dict:
    from graph_rag.graphrag_engine import MSGraphRAGEngine
    from django.conf import settings

    if not getattr(settings, "OPENAI_API_KEY", ""):
        print("Warning: OPENAI_API_KEY is not set in settings/.env; query may fail.")

    engine = MSGraphRAGEngine(project_name=project_name)
    result = await engine.query_graphrag(query_text=query_text, query_type=query_type)

    print("=== GraphRAG Query Result ===")
    print(json.dumps({k: v for k, v in result.items() if k != "response"}, indent=2))
    # Print only a small slice of stdout to avoid flooding the console
    stdout = result.get("response", "") or ""
    if stdout:
        print("\n=== GraphRAG stdout (first 40 lines) ===")
        lines = stdout.splitlines()
        for line in lines[:40]:
            print(line)

    # Attempt to read common GraphRAG output artifacts if present
    print("\n=== Inspecting output artifacts ===")
    read_outputs(Path(engine.working_dir) / "output")

    return result


def read_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    context_path = output_dir / "context.json"
    if context_path.exists():
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                context = json.load(f)
            print("Loaded context.json with keys:", list(context.keys()))
        except Exception as exc:
            print(f"Failed to read context.json: {exc}")
    else:
        print("Missing context.json")


def main() -> None:
    setup_django()

    # You can change the project name to isolate outputs per run
    project_name = os.environ.get("GRAPH_RAG_TEST_PROJECT", "example-test")
    # Adjust the query as needed
    query_text = os.environ.get("GRAPH_RAG_TEST_QUERY", "who are the main characters and communities?")
    query_type = os.environ.get("GRAPH_RAG_TEST_METHOD", "local")  # e.g., "local", "global"

    asyncio.run(run_query(project_name=project_name, query_text=query_text, query_type=query_type))


if __name__ == "__main__":
    main()

