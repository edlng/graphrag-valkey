# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Registration helpers for plugging Valkey backends into GraphRAG factories."""

import logging

from graphrag_cache import register_cache as _register_cache
from graphrag_storage import register_storage as _register_storage
from graphrag_vectors import register_vector_store as _register_vector_store

from graphrag_valkey.cache import ValkeyCache
from graphrag_valkey.storage import ValkeyStorage
from graphrag_valkey.vector_store import ValkeyVectorStore

logger = logging.getLogger(__name__)

VALKEY_TYPE = "valkey"


def register_storage() -> None:
    """Register ValkeyStorage with the GraphRAG storage factory."""
    _register_storage(VALKEY_TYPE, ValkeyStorage)
    logger.info("Registered Valkey storage backend")


def register_cache() -> None:
    """Register ValkeyCache with the GraphRAG cache factory."""
    _register_cache(VALKEY_TYPE, ValkeyCache)
    logger.info("Registered Valkey cache backend")


def register_vector_store() -> None:
    """Register ValkeyVectorStore with the GraphRAG vector store factory."""
    _register_vector_store(VALKEY_TYPE, ValkeyVectorStore)
    logger.info("Registered Valkey vector store backend")


def register_all() -> None:
    """Register all Valkey backends with their respective GraphRAG factories.

    Call this once at application startup before creating any storage,
    cache, or vector store instances.

    Example
    -------
        from graphrag_valkey import register_all
        register_all()
    """
    register_storage()
    register_cache()
    register_vector_store()
