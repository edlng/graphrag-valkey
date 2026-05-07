# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Shared fixtures for graphrag-valkey integration tests.

Provides a session-scoped Valkey connectivity check and reusable client fixture.
All integration tests are skipped if Valkey is not reachable on localhost:6379.
"""

import pytest
from glide_sync import GlideClient, GlideClientConfiguration, NodeAddress

from graphrag_valkey import register_all

VALKEY_HOST = "localhost"
VALKEY_PORT = 6379


def _is_valkey_available() -> bool:
    """Check if Valkey is reachable. Connects and immediately closes."""
    try:
        config = GlideClientConfiguration(
            addresses=[NodeAddress(VALKEY_HOST, VALKEY_PORT)],
            request_timeout=2000,
        )
        client = GlideClient.create(config)
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "valkey: mark test as requiring a running Valkey instance"
    )


def pytest_collection_modifyitems(config, items):
    """Skip all integration tests if Valkey is not available."""
    if not _is_valkey_available():
        skip_marker = pytest.mark.skip(
            reason="Valkey not available on localhost:6379"
        )
        for item in items:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def _register_backends():
    """Register all Valkey backends once per test session."""
    register_all()


@pytest.fixture
def valkey_client():
    """Provide a connected GlideClient for tests that need direct access.

    The client is closed after each test.
    """
    config = GlideClientConfiguration(
        addresses=[NodeAddress(VALKEY_HOST, VALKEY_PORT)],
        request_timeout=2000,
        database_id=0,
    )
    client = GlideClient.create(config)
    yield client
    client.close()
