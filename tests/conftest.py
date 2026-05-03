"""Pytest configuration: ensure app.state.model is populated before tests run.

The existing test_api.py uses a module-level TestClient(app) without a context
manager, so the FastAPI lifespan never fires during tests. This conftest loads
the model into app.state directly so all tests can access it.
"""

import pytest

from app.main import app
from app.model import load_model


@pytest.fixture(scope="session", autouse=True)
def load_model_into_app_state():
    """Load the model into app.state once for the entire test session."""
    app.state.model = load_model()
