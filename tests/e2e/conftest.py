# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Shared fixtures for graphrag-valkey E2E tests.

E2E tests require:
- A running Valkey instance with valkey-search module on localhost:6379
- Ollama running on localhost:11434 with models:
    ollama pull gemma4:e2b
    ollama pull nomic-embed-text
- The graphrag package installed

All E2E tests are skipped if Valkey is not reachable.
"""

import pytest
from glide_sync import GlideClient, GlideClientConfiguration, NodeAddress

from graphrag_valkey import register_all

VALKEY_HOST = "localhost"
VALKEY_PORT = 6379


def _is_valkey_available() -> bool:
    """Check if Valkey is reachable."""
    try:
        config = GlideClientConfiguration(
            addresses=[NodeAddress(VALKEY_HOST, VALKEY_PORT)],
            request_timeout=3000,
        )
        client = GlideClient.create(config)
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test requiring Valkey and Ollama"
    )


def pytest_collection_modifyitems(config, items):
    """Skip all E2E tests if Valkey is not available."""
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


@pytest.fixture(autouse=True)
def _cleanup_valkey():
    """Clean up E2E test data from Valkey after each test.

    Uses prefix-based SCAN + DELETE to remove only test keys,
    avoiding FLUSHDB which would be too destructive on shared instances.
    """
    yield
    for db in (0, 1, 2):
        try:
            config = GlideClientConfiguration(
                addresses=[NodeAddress(VALKEY_HOST, VALKEY_PORT)],
                request_timeout=5000,
                database_id=db,
            )
            client = GlideClient.create(config)
            for prefix in ("e2e_output:", "e2e_cache:", "e2e_vs:", "idx:e2e_vs:"):
                cursor = "0"
                while True:
                    result = client.scan(cursor, match=f"{prefix}*")
                    keys = result[1]
                    if keys:
                        client.delete(keys)
                    cursor_bytes = result[0]
                    cursor_str = (
                        cursor_bytes.decode()
                        if isinstance(cursor_bytes, bytes)
                        else str(cursor_bytes)
                    )
                    if cursor_str == "0":
                        break
                    cursor = cursor_str
            client.close()
        except Exception:  # noqa: BLE001
            pass
