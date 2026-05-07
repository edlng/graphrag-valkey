# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Integration tests for ValkeyCache.

Requires a running Valkey instance on localhost:6379.
"""

from graphrag_cache import CacheConfig, create_cache


class TestValkeyCache:
    """Tests for ValkeyCache via the factory."""

    async def test_set_get_delete(self):
        cache = create_cache(
            CacheConfig(type="valkey", host="localhost", port=6379, db=1, prefix="test_cache")
        )
        try:
            await cache.set("ck", "cv")
            assert await cache.get("ck") == "cv"
            assert await cache.has("ck") is True

            await cache.delete("ck")
            assert await cache.has("ck") is False
        finally:
            await cache.close()

    async def test_set_none_is_noop(self):
        cache = create_cache(
            CacheConfig(type="valkey", host="localhost", port=6379, db=1, prefix="test_noop")
        )
        try:
            await cache.set("nk", None)
            assert await cache.has("nk") is False
        finally:
            await cache.close()

    async def test_clear(self):
        cache = create_cache(
            CacheConfig(type="valkey", host="localhost", port=6379, db=1, prefix="test_cclear")
        )
        try:
            await cache.set("x", "1")
            await cache.set("y", "2")
            await cache.clear()
            assert await cache.has("x") is False
            assert await cache.has("y") is False
        finally:
            await cache.close()

    async def test_child(self):
        cache = create_cache(
            CacheConfig(type="valkey", host="localhost", port=6379, db=1, prefix="test_cparent")
        )
        try:
            child = cache.child("sub")
            await child.set("ck", "cv")
            assert await child.get("ck") == "cv"
            assert await cache.has("ck") is False
            await child.clear()
            await child.close()
        finally:
            await cache.close()

    async def test_debug_data(self):
        cache = create_cache(
            CacheConfig(type="valkey", host="localhost", port=6379, db=1, prefix="test_debug")
        )
        try:
            await cache.set("dk", "dv", debug_data={"extra": "info"})
            assert await cache.get("dk") == "dv"
            await cache.clear()
        finally:
            await cache.close()
