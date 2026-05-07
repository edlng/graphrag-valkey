# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Unit tests for ValkeyStorage — key prefixing, stripping, and metadata key logic."""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_valkey.storage import ValkeyStorage


class TestKeyPrefixing:
    """Tests for key prefix/strip logic (no Valkey connection needed)."""

    def test_prefixed_key_with_prefix(self):
        storage = ValkeyStorage(prefix="myprefix")
        assert storage._prefixed_key("doc1") == "myprefix:doc1"

    def test_prefixed_key_default_prefix(self):
        storage = ValkeyStorage()
        assert storage._prefixed_key("doc1") == "graphrag:doc1"

    def test_prefixed_key_without_prefix(self):
        storage = ValkeyStorage(prefix="")
        assert storage._prefixed_key("doc1") == "doc1"

    def test_strip_prefix_with_prefix(self):
        storage = ValkeyStorage(prefix="myprefix")
        assert storage._strip_prefix("myprefix:doc1") == "doc1"

    def test_strip_prefix_without_prefix(self):
        storage = ValkeyStorage(prefix="")
        assert storage._strip_prefix("doc1") == "doc1"

    def test_strip_prefix_no_match(self):
        storage = ValkeyStorage(prefix="myprefix")
        # Key doesn't start with expected prefix — returned as-is
        assert storage._strip_prefix("other:doc1") == "other:doc1"

    def test_meta_key_with_prefix(self):
        storage = ValkeyStorage(prefix="pfx")
        assert storage._meta_key("k") == "pfx:k:__meta__"

    def test_meta_key_without_prefix(self):
        storage = ValkeyStorage(prefix="")
        assert storage._meta_key("k") == "k:__meta__"


class TestChildStorage:
    """Tests for child() prefix composition."""

    def test_child_extends_prefix(self):
        storage = ValkeyStorage(prefix="root")
        child = storage.child("sub")
        assert child._prefix == "root:sub"

    def test_child_of_no_prefix(self):
        storage = ValkeyStorage(prefix="")
        child = storage.child("sub")
        assert child._prefix == "sub"

    def test_child_of_default_prefix(self):
        storage = ValkeyStorage()
        child = storage.child("sub")
        assert child._prefix == "graphrag:sub"

    def test_child_none_returns_self(self):
        storage = ValkeyStorage(prefix="root")
        assert storage.child(None) is storage

    def test_child_inherits_connection_params(self):
        storage = ValkeyStorage(host="myhost", port=1234, db=5, prefix="root")
        child = storage.child("sub")
        assert child._host == "myhost"
        assert child._port == 1234
        assert child._db == 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(prefix: str = "pfx") -> tuple[ValkeyStorage, AsyncMock]:
    """Return a ValkeyStorage with a pre-injected async mock client."""
    storage = ValkeyStorage(prefix=prefix)
    mock_client = AsyncMock()
    storage._async_client = mock_client
    return storage, mock_client


def _make_sync_storage(prefix: str = "pfx") -> tuple[ValkeyStorage, MagicMock]:
    """Return a ValkeyStorage with a pre-injected sync mock client."""
    storage = ValkeyStorage(prefix=prefix)
    mock_client = MagicMock()
    storage._sync_client = mock_client
    return storage, mock_client


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestStorageGet:
    async def test_returns_decoded_string(self):
        storage, client = _make_storage()
        client.get.return_value = b"hello"
        result = await storage.get("mykey")
        assert result == "hello"
        client.get.assert_called_once_with("pfx:mykey")

    async def test_returns_none_when_missing(self):
        storage, client = _make_storage()
        client.get.return_value = None
        assert await storage.get("missing") is None

    async def test_returns_bytes_when_as_bytes(self):
        storage, client = _make_storage()
        client.get.return_value = b"rawdata"
        result = await storage.get("k", as_bytes=True)
        assert result == b"rawdata"

    async def test_encodes_string_result_to_bytes_when_as_bytes(self):
        storage, client = _make_storage()
        client.get.return_value = "strval"
        result = await storage.get("k", as_bytes=True)
        assert result == b"strval"

    async def test_returns_none_on_exception(self):
        storage, client = _make_storage()
        client.get.side_effect = RuntimeError("connection lost")
        assert await storage.get("k") is None


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------

class TestStorageSet:
    async def test_stores_string_as_bytes(self):
        storage, client = _make_storage()
        await storage.set("doc1", "content")
        client.set.assert_any_call("pfx:doc1", b"content")

    async def test_stores_bytes_directly(self):
        storage, client = _make_storage()
        await storage.set("doc1", b"\x00\x01")
        client.set.assert_any_call("pfx:doc1", b"\x00\x01")

    async def test_stores_non_string_as_str_bytes(self):
        storage, client = _make_storage()
        await storage.set("doc1", 42)
        client.set.assert_any_call("pfx:doc1", b"42")

    async def test_writes_metadata_key(self):
        storage, client = _make_storage()
        await storage.set("doc1", "v")
        assert client.set.call_args_list[1].args[0] == "pfx:doc1:__meta__"

    async def test_silently_handles_exception(self):
        storage, client = _make_storage()
        client.set.side_effect = RuntimeError("write error")
        # Should not raise
        await storage.set("k", "v")


# ---------------------------------------------------------------------------
# has()
# ---------------------------------------------------------------------------

class TestStorageHas:
    async def test_returns_true_when_key_exists(self):
        storage, client = _make_storage()
        client.exists.return_value = 1
        assert await storage.has("doc1") is True
        client.exists.assert_called_once_with(["pfx:doc1"])

    async def test_returns_false_when_key_missing(self):
        storage, client = _make_storage()
        client.exists.return_value = 0
        assert await storage.has("doc1") is False

    async def test_returns_false_on_exception(self):
        storage, client = _make_storage()
        client.exists.side_effect = RuntimeError("timeout")
        assert await storage.has("k") is False


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestStorageDelete:
    async def test_deletes_key_and_meta(self):
        storage, client = _make_storage()
        await storage.delete("doc1")
        deleted = [call.args[0] for call in client.delete.call_args_list]
        assert ["pfx:doc1"] in deleted
        assert ["pfx:doc1:__meta__"] in deleted

    async def test_silently_handles_exception(self):
        storage, client = _make_storage()
        client.delete.side_effect = RuntimeError("gone")
        await storage.delete("k")  # should not raise


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestStorageClear:
    async def test_deletes_all_prefixed_keys(self):
        storage, client = _make_storage(prefix="pfx")
        # SCAN returns one page then done
        client.scan.return_value = (b"0", [b"pfx:a", b"pfx:b"])
        await storage.clear()
        client.delete.assert_called_once()
        deleted_keys = client.delete.call_args.args[0]
        assert "pfx:a" in deleted_keys
        assert "pfx:b" in deleted_keys

    async def test_no_delete_when_no_keys(self):
        storage, client = _make_storage(prefix="pfx")
        client.scan.return_value = (b"0", [])
        await storage.clear()
        client.delete.assert_not_called()

    async def test_warns_and_skips_when_no_prefix(self, caplog):
        storage, client = _make_storage(prefix="")
        await storage.clear()
        client.delete.assert_not_called()
        assert "refusing to flush" in caplog.text

    async def test_silently_handles_exception(self):
        storage, client = _make_storage()
        client.scan.side_effect = RuntimeError("scan failed")
        await storage.clear()  # should not raise


# ---------------------------------------------------------------------------
# find() / keys()
# ---------------------------------------------------------------------------

class TestStorageFindAndKeys:
    def test_find_yields_matching_keys(self):
        storage, client = _make_sync_storage(prefix="pfx")
        client.scan.return_value = (b"0", [b"pfx:report.json", b"pfx:data.csv"])
        results = list(storage.find(re.compile(r"\.json$")))
        assert results == ["report.json"]

    def test_find_skips_metadata_keys(self):
        storage, client = _make_sync_storage(prefix="pfx")
        client.scan.return_value = (b"0", [b"pfx:doc1", b"pfx:doc1:__meta__"])
        results = list(storage.find(re.compile(r".*")))
        assert "doc1:__meta__" not in results
        assert "doc1" in results

    def test_keys_returns_stripped_keys(self):
        storage, client = _make_sync_storage(prefix="pfx")
        client.scan.return_value = (b"0", [b"pfx:a", b"pfx:b", b"pfx:b:__meta__"])
        result = storage.keys()
        assert sorted(result) == ["a", "b"]

    def test_find_paginates_scan(self):
        storage, client = _make_sync_storage(prefix="pfx")
        # First call returns cursor "1", second returns "0"
        client.scan.side_effect = [
            (b"1", [b"pfx:first.txt"]),
            (b"0", [b"pfx:second.txt"]),
        ]
        results = list(storage.find(re.compile(r".*")))
        assert len(results) == 2
        assert client.scan.call_count == 2


# ---------------------------------------------------------------------------
# get_creation_date()
# ---------------------------------------------------------------------------

class TestStorageGetCreationDate:
    async def test_returns_formatted_date(self):
        storage, client = _make_storage()
        # Use a known timestamp
        client.get.return_value = b"0.0"  # epoch
        result = await storage.get_creation_date("doc1")
        assert isinstance(result, str)
        assert result != ""

    async def test_returns_empty_when_no_meta(self):
        storage, client = _make_storage()
        client.get.return_value = None
        assert await storage.get_creation_date("doc1") == ""

    async def test_returns_empty_on_exception(self):
        storage, client = _make_storage()
        client.get.side_effect = RuntimeError("gone")
        assert await storage.get_creation_date("doc1") == ""


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestStorageClose:
    async def test_closes_async_client(self):
        storage, async_client = _make_storage()
        await storage.close()
        async_client.close.assert_called_once()
        assert storage._async_client is None

    async def test_closes_sync_client(self):
        storage, _ = _make_storage()
        sync_client = MagicMock()
        storage._sync_client = sync_client
        await storage.close()
        sync_client.close.assert_called_once()
        assert storage._sync_client is None
