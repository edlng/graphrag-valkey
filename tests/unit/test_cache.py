# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Unit tests for ValkeyCache — key prefixing, child logic, and I/O methods."""

import json
from unittest.mock import AsyncMock

from graphrag_valkey.cache import ValkeyCache


class TestCacheKeyPrefixing:
    """Tests for cache key prefix logic (no Valkey connection needed)."""

    def test_prefixed_key(self):
        cache = ValkeyCache(prefix="test_cache")
        assert cache._prefixed_key("mykey") == "test_cache:mykey"

    def test_prefixed_key_default_prefix(self):
        cache = ValkeyCache()
        assert cache._prefixed_key("k") == "graphrag_cache:k"


class TestCacheChild:
    """Tests for child() prefix composition."""

    def test_child_extends_prefix(self):
        cache = ValkeyCache(prefix="parent")
        child = cache.child("sub")
        assert child._prefix == "parent:sub"

    def test_child_inherits_connection_params(self):
        cache = ValkeyCache(host="myhost", port=9999, db=3, prefix="p")
        child = cache.child("c")
        assert child._host == "myhost"
        assert child._port == 9999
        assert child._db == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(prefix: str = "c") -> tuple[ValkeyCache, AsyncMock]:
    """Return a ValkeyCache with a pre-injected async mock client."""
    cache = ValkeyCache(prefix=prefix)
    mock_client = AsyncMock()
    cache._client = mock_client
    return cache, mock_client


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestCacheGet:
    async def test_returns_result_value(self):
        cache, client = _make_cache()
        payload = json.dumps({"result": "cached_answer"}).encode()
        client.get.return_value = payload
        assert await cache.get("k") == "cached_answer"
        client.get.assert_called_once_with("c:k")

    async def test_returns_none_when_missing(self):
        cache, client = _make_cache()
        client.get.return_value = None
        assert await cache.get("k") is None

    async def test_returns_none_and_deletes_on_corrupt_json(self):
        cache, client = _make_cache()
        client.get.return_value = b"not-json"
        result = await cache.get("k")
        assert result is None
        client.delete.assert_called_once_with(["c:k"])

    async def test_returns_none_on_exception(self):
        cache, client = _make_cache()
        client.get.side_effect = RuntimeError("timeout")
        assert await cache.get("k") is None

    async def test_returns_complex_value(self):
        cache, client = _make_cache()
        payload = json.dumps({"result": {"nested": [1, 2, 3]}}).encode()
        client.get.return_value = payload
        assert await cache.get("k") == {"nested": [1, 2, 3]}


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------

class TestCacheSet:
    async def test_stores_serialized_result(self):
        cache, client = _make_cache()
        await cache.set("k", "myvalue")
        client.set.assert_called_once()
        key, raw = client.set.call_args.args
        assert key == "c:k"
        stored = json.loads(raw)
        assert stored["result"] == "myvalue"

    async def test_skips_none_value(self):
        cache, client = _make_cache()
        await cache.set("k", None)
        client.set.assert_not_called()

    async def test_stores_debug_data(self):
        cache, client = _make_cache()
        await cache.set("k", "v", debug_data={"tokens": 42})
        _, raw = client.set.call_args.args
        stored = json.loads(raw)
        assert stored["debug"] == {"tokens": 42}

    async def test_silently_handles_exception(self):
        cache, client = _make_cache()
        client.set.side_effect = RuntimeError("write error")
        await cache.set("k", "v")  # should not raise


# ---------------------------------------------------------------------------
# has()
# ---------------------------------------------------------------------------

class TestCacheHas:
    async def test_returns_true_when_key_exists(self):
        cache, client = _make_cache()
        client.exists.return_value = 1
        assert await cache.has("k") is True
        client.exists.assert_called_once_with(["c:k"])

    async def test_returns_false_when_key_missing(self):
        cache, client = _make_cache()
        client.exists.return_value = 0
        assert await cache.has("k") is False

    async def test_returns_false_on_exception(self):
        cache, client = _make_cache()
        client.exists.side_effect = RuntimeError("gone")
        assert await cache.has("k") is False


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestCacheDelete:
    async def test_deletes_prefixed_key(self):
        cache, client = _make_cache()
        await cache.delete("k")
        client.delete.assert_called_once_with(["c:k"])

    async def test_silently_handles_exception(self):
        cache, client = _make_cache()
        client.delete.side_effect = RuntimeError("gone")
        await cache.delete("k")  # should not raise


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestCacheClear:
    async def test_deletes_all_prefixed_keys(self):
        cache, client = _make_cache(prefix="c")
        client.scan.return_value = (b"0", [b"c:a", b"c:b"])
        await cache.clear()
        client.delete.assert_called_once()
        deleted = client.delete.call_args.args[0]
        assert "c:a" in deleted
        assert "c:b" in deleted

    async def test_no_delete_when_no_keys(self):
        cache, client = _make_cache()
        client.scan.return_value = (b"0", [])
        await cache.clear()
        client.delete.assert_not_called()

    async def test_paginates_scan(self):
        cache, client = _make_cache(prefix="c")
        client.scan.side_effect = [
            (b"1", [b"c:first"]),
            (b"0", [b"c:second"]),
        ]
        await cache.clear()
        assert client.scan.call_count == 2

    async def test_silently_handles_exception(self):
        cache, client = _make_cache()
        client.scan.side_effect = RuntimeError("scan failed")
        await cache.clear()  # should not raise


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestCacheClose:
    async def test_closes_client(self):
        cache, client = _make_cache()
        await cache.close()
        client.close.assert_called_once()
        assert cache._client is None

    async def test_close_is_idempotent(self):
        cache, client = _make_cache()
        await cache.close()
        await cache.close()  # second call should not raise
        client.close.assert_called_once()
