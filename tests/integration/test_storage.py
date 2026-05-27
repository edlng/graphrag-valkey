# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Integration tests for ValkeyStorage.

Requires a running Valkey instance on localhost:6379.
"""

import re

from graphrag_storage import StorageConfig, create_storage


class TestValkeyStorage:
    """Tests for ValkeyStorage via the factory."""

    async def test_set_get_delete(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_storage")
        )
        try:
            await storage.set("key1", "value1")
            assert await storage.get("key1") == "value1"
            assert await storage.has("key1") is True

            await storage.delete("key1")
            assert await storage.has("key1") is False
            assert await storage.get("key1") is None
        finally:
            await storage.close()

    async def test_set_get_bytes(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_bytes")
        )
        try:
            await storage.set("binkey", b"\x00\x01\x02")
            result = await storage.get("binkey", as_bytes=True)
            assert result == b"\x00\x01\x02"
            await storage.clear()
        finally:
            await storage.close()

    async def test_clear(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_clear")
        )
        try:
            await storage.set("a", "1")
            await storage.set("b", "2")
            await storage.clear()
            assert await storage.has("a") is False
            assert await storage.has("b") is False
        finally:
            await storage.close()

    async def test_child(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_parent")
        )
        try:
            child = storage.child("sub")
            await child.set("ck", "cv")
            assert await child.get("ck") == "cv"

            # Parent should not see child key without prefix
            assert await storage.has("ck") is False
            await child.clear()
            await child.close()
        finally:
            await storage.close()

    async def test_get_creation_date(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_date")
        )
        try:
            await storage.set("dated", "val")
            date_str = await storage.get_creation_date("dated")
            assert date_str != ""
            assert len(date_str) > 0
            await storage.clear()
        finally:
            await storage.close()

    async def test_find(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_find")
        )
        try:
            await storage.set("report.txt", "data1")
            await storage.set("report.json", "data2")
            await storage.set("other.txt", "data3")

            results = list(storage.find(re.compile(r"\.txt$")))
            assert sorted(results) == ["other.txt", "report.txt"]

            results = list(storage.find(re.compile(r"^report")))
            assert sorted(results) == ["report.json", "report.txt"]

            results = list(storage.find(re.compile(r"^nonexistent")))
            assert results == []

            await storage.clear()
        finally:
            await storage.close()

    async def test_keys(self):
        storage = create_storage(
            StorageConfig(type="valkey", host="localhost", port=6379, db=0, prefix="test_keys")
        )
        try:
            await storage.set("alpha", "1")
            await storage.set("beta", "2")
            await storage.set("gamma", "3")

            result = storage.keys()
            assert sorted(result) == ["alpha", "beta", "gamma"]

            await storage.delete("beta")
            result = storage.keys()
            assert sorted(result) == ["alpha", "gamma"]

            await storage.clear()
        finally:
            await storage.close()
