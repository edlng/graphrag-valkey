# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Factory-path smoke tests for graphrag-valkey.

The storage and cache integration tests already drive the graphrag factory
path end-to-end (see test_storage.py, test_cache.py). This file only covers
what they don't:

1. After register_all(), each factory resolves "valkey" to the correct class.
2. The vector store factory path works (test_vector_store.py constructs the
   store directly, bypassing create_vector_store).

Requires a running Valkey instance with the valkey-search module on localhost:6379.
"""

import uuid

import pytest
from graphrag_cache import CacheConfig, create_cache
from graphrag_storage import StorageConfig, create_storage
from graphrag_vectors import (
    IndexSchema,
    VectorStoreConfig,
    VectorStoreDocument,
    create_vector_store,
)

from graphrag_valkey import ValkeyCache, ValkeyStorage, ValkeyVectorStore


def _unique_prefix() -> str:
    """Generate a unique prefix to isolate test runs."""
    return f"test_factory_{uuid.uuid4().hex[:8]}"


def test_storage_factory_resolves_valkey():
    """create_storage should return a ValkeyStorage instance for type='valkey'."""
    s = create_storage(
        StorageConfig(
            type="valkey", host="localhost", port=6379, db=0, prefix=_unique_prefix()
        )
    )
    assert isinstance(s, ValkeyStorage)


def test_cache_factory_resolves_valkey():
    """create_cache should return a ValkeyCache instance for type='valkey'."""
    c = create_cache(
        CacheConfig(
            type="valkey",
            host="localhost",
            port=6379,
            db=1,
            prefix=_unique_prefix(),
            storage=None,
        )
    )
    assert isinstance(c, ValkeyCache)


class TestVectorStoreFactory:
    """Exercise the create_vector_store(config, schema) factory path.

    Other tests (test_vector_store.py) instantiate ValkeyVectorStore directly;
    this class makes sure the config+schema merge in create_vector_store
    produces a working store.
    """

    @pytest.fixture
    def vector_store(self):
        config = VectorStoreConfig(
            type="valkey",
            host="localhost",
            port=6379,
            db=0,
            prefix=_unique_prefix(),
        )
        schema = IndexSchema(
            index_name="factory_test",
            vector_size=5,
            fields={"category": "str"},
        )
        vs = create_vector_store(config, schema)
        vs.connect()
        vs.create_index()
        yield vs
        vs.drop_index()
        vs.close()

    def test_factory_returns_valkey_vector_store(self, vector_store):
        assert isinstance(vector_store, ValkeyVectorStore)

    def test_factory_instance_round_trips_a_document(self, vector_store):
        """Sanity check that the factory-built store can load and retrieve."""
        vector_store.load_documents([
            VectorStoreDocument(
                id="doc-1",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                data={"category": "entity"},
            ),
        ])
        doc = vector_store.search_by_id("doc-1")
        assert doc.id == "doc-1"
        assert doc.data["category"] == "entity"
