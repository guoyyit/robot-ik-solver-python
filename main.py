"""Application entry point: ``uv run uvicorn main:app``."""

from ik_service.api import create_app

app = create_app()
