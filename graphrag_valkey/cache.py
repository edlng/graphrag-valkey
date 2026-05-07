# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Valkey Cache implementation for GraphRAG."""

import asyncio
import json
import logging
from typing import Any

from glide import GlideClient
from graphrag_cache.cache import Cache

from graphrag_valkey.client import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
    KEY_SEPARATOR,
    create_async_client,
    scan_keys_async,
)

logger = logging.getLogger(__name__)


class ValkeyCache(Cache):
    """Valkey-backed cache implementation using the async valkey-glide client."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        db: int = DEFAULT_DB,
        prefix: str = "graphrag_cache",
        encoding: str = "utf-8",
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        # Accept and ignore storage kwarg from the factory pattern
        storage: Any = None,
        **kwargs: Any,
    ) -> None:
        """Create a Valkey cache instance."""
        super().__init__(storage=storage, **kwargs)
        self._host = host
        self._port = port
        self._db = db
        self._prefix = prefix
        self._encoding = encoding
        self._request_timeout = request_timeout
        self._client: GlideClient | None = None
        self._client_lock = asyncio.Lock()
        logger.info(
            "Creating Valkey cache at %s:%d (db=%d, prefix='%s')",
            self._host,
            self._port,
            self._db,
            self._prefix,
        )

    async def _get_client(self) -> GlideClient:
        """Lazily initialize and return the async Valkey client."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = await create_async_client(
                        self._host, self._port, self._db, self._request_timeout
                    )
        return self._client

    async def close(self) -> None:
        """Close the underlying Valkey client connection."""
        async with self._client_lock:
            if self._client is not None:
                await self._client.close()
                self._client = None

    def _prefixed_key(self, key: str) -> str:
        """Return the full key with prefix applied."""
        return f"{self._prefix}{KEY_SEPARATOR}{key}"

    async def get(self, key: str) -> Any | None:
        """Get the cached value for the given key."""
        try:
            client = await self._get_client()
            result = await client.get(self._prefixed_key(key))
            if result is None:
                return None
            raw = result.decode(self._encoding) if isinstance(result, bytes) else str(result)
            data = json.loads(raw)
            return data.get("result")
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Corrupt cache entry for key %s, deleting", key)
            await self.delete(key)
            return None
        except Exception:
            logger.exception("Error reading cache key %s from Valkey", key)
            return None

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set the cached value for the given key."""
        if value is None:
            return
        try:
            client = await self._get_client()
            data: dict[str, Any] = {"result": value}
            if debug_data:
                data["debug"] = debug_data
            serialized = json.dumps(data, ensure_ascii=False).encode(self._encoding)
            await client.set(self._prefixed_key(key), serialized)
        except Exception:
            logger.exception("Error writing cache key %s to Valkey", key)

    async def has(self, key: str) -> bool:
        """Return True if the given key exists in the cache."""
        try:
            client = await self._get_client()
            count = await client.exists([self._prefixed_key(key)])
            return count > 0  # noqa: TRY300
        except Exception:
            logger.exception("Error checking cache key %s in Valkey", key)
            return False

    async def delete(self, key: str) -> None:
        """Delete the given key from the cache."""
        try:
            client = await self._get_client()
            await client.delete([self._prefixed_key(key)])
        except Exception:
            logger.exception("Error deleting cache key %s from Valkey", key)

    async def clear(self) -> None:
        """Clear all cache keys under the current prefix."""
        try:
            client = await self._get_client()
            pattern = f"{self._prefix}{KEY_SEPARATOR}*"
            all_keys = await scan_keys_async(client, pattern, self._encoding)
            if all_keys:
                await client.delete(all_keys)
        except Exception:
            logger.exception("Error clearing Valkey cache")

    def child(self, name: str) -> "Cache":
        """Create a child cache with the given name."""
        child_prefix = f"{self._prefix}{KEY_SEPARATOR}{name}"
        return ValkeyCache(
            host=self._host,
            port=self._port,
            db=self._db,
            prefix=child_prefix,
            encoding=self._encoding,
            request_timeout=self._request_timeout,
        )
