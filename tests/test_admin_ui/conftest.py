"""Conftest for UI tests — overrides async autouse from parent conftest."""
import pytest


@pytest.fixture(autouse=True)
def _cleanup_after_test():
    """Shadow conftest.py's async autouse fixture with sync no-op."""
    yield
