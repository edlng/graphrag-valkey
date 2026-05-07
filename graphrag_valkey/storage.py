# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Valkey Storage implementation for GraphRAG."""

import asyncio
import logging
import re
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from glide import GlideClient as AsyncGlideClient
from glide_sync import GlideClient as SyncGlideClient
from graphrag_storage.storage import (
    Storage,
    get_timestamp_formatted_with_local_tz,
)

from graphrag_valkey.client import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
    KEY_SEPARATOR,
    METADATA_SUFFIX,
    create_async_client,
    create_sync_client,
    scan_keys_async,
    scan_keys_sync,
)

logger = logging.getLogger(__name__)


class ValkeyStorage(Storage):
    """Valkey-backed storage implementation.

    Async methods use the async valkey-glide client for non-blocking I/O.
    Sync methods (find, keys) use the sync valkey-glide-sync client.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        db: int = DEFAULT_DB,
        prefix: str = "graphrag",
        encoding: str = "utf-8",
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        """Create a Valkey storage instance."""
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._db = db
        self._prefix = prefix
        self._encoding = encoding
        self._request_timeout = request_timeout
        self._async_client: AsyncGlideClient | None = None
        self._async_lock = asyncio.Lock()
        self._sync_client: SyncGlideClient | None = None
        self._sync_lock = threading.Lock()
        logger.info(
            "Creating Valkey storage at %s:%d (db=%d, prefix='%s')",
            self._host,
            self._port,
            self._db,
            self._prefix,
        )

    async def _get_async_client(self) -> AsyncGlideClient:
        """Lazily initialize and return the async Valkey client."""
        if self._async_client is None:
            async with self._async_lock:
                if self._async_client is None:
                    self._async_client = await create_async_client(
                        self._host, self._port, self._db, self._request_timeout
                    )
        return self._async_client

    def _get_sync_client(self) -> SyncGlideClient:
        """Lazily initialize and return the sync Valkey client."""
        if self._sync_client is None:
            with self._sync_lock:
                if self._sync_client is None:
                    self._sync_client = create_sync_client(
                        self._host, self._port, self._db, self._request_timeout
                    )
        return self._sync_client

    async def close(self) -> None:
        """Close the underlying Valkey client connections."""
        async with self._async_lock:
            if self._async_client is not None:
                await self._async_client.close()
                self._async_client = None
        with self._sync_lock:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None

    def _prefixed_key(self, key: str) -> str:
        """Return the full key with prefix applied."""
        if self._prefix:
            return f"{self._prefix}{KEY_SEPARATOR}{key}"
        return key

    def _strip_prefix(self, full_key: str) -> str:
        """Strip the prefix from a full key."""
        if self._prefix:
            expected = f"{self._prefix}{KEY_SEPARATOR}"
            if full_key.startswith(expected):
                return full_key[len(expected):]
        return full_key

    def _meta_key(self, key: str) -> str:
        """Return the metadata key for a given storage key."""
        return f"{self._prefixed_key(key)}{METADATA_SUFFIX}"

    def find(
        self,
        file_pattern: re.Pattern[str],
    ) -> Iterator[str]:
        """Find keys matching the given regex pattern."""
        logger.info(
            "Searching Valkey for keys matching [%s] (prefix='%s')",
            file_pattern.pattern,
            self._prefix,
        )
        prefix_pattern = (
            f"{self._prefix}{KEY_SEPARATOR}*" if self._prefix else "*"
        )

        all_keys = scan_keys_sync(self._get_sync_client(), prefix_pattern, self._encoding)

        for full_key in all_keys:
            if full_key.endswith(METADATA_SUFFIX):
                continue
            stripped = self._strip_prefix(full_key)
            if file_pattern.search(stripped):
                yield stripped

    async def get(
        self, key: str, as_bytes: bool | None = None, encoding: str | None = None
    ) -> Any:
        """Get the value for the given key."""
        try:
            client = await self._get_async_client()
            result = await client.get(self._prefixed_key(key))
            if result is None:
                return None
            if as_bytes:
                return result if isinstance(result, bytes) else result.encode(
                    encoding or self._encoding
                )
            if isinstance(result, bytes):
                return result.decode(encoding or self._encoding)
            return result  # noqa: TRY300
        except Exception:
            logger.exception("Error reading key %s from Valkey", key)
            return None

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set the value for the given key."""
        try:
            client = await self._get_async_client()
            if isinstance(value, bytes):
                store_value = value
            elif isinstance(value, str):
                store_value = value.encode(encoding or self._encoding)
            else:
                store_value = str(value).encode(encoding or self._encoding)

            await client.set(self._prefixed_key(key), store_value)

            # Store creation timestamp as metadata
            meta = str(datetime.now(tz=timezone.utc).timestamp())
            await client.set(self._meta_key(key), meta.encode(self._encoding))
        except Exception:
            logger.exception("Error writing key %s to Valkey", key)

    async def has(self, key: str) -> bool:
        """Return True if the given key exists."""
        try:
            client = await self._get_async_client()
            count = await client.exists([self._prefixed_key(key)])
            return count > 0  # noqa: TRY300
        except Exception:
            logger.exception("Error checking existence of key %s in Valkey", key)
            return False

    async def delete(self, key: str) -> None:
        """Delete the given key."""
        try:
            client = await self._get_async_client()
            await client.delete([self._prefixed_key(key)])
            await client.delete([self._meta_key(key)])
        except Exception:
            logger.exception("Error deleting key %s from Valkey", key)

    async def clear(self) -> None:
        """Clear all keys under the current prefix.

        If no prefix is set, logs a warning and does nothing to avoid
        accidentally flushing the entire database.
        """
        try:
            if not self._prefix:
                logger.warning(
                    "clear() called with no prefix set — refusing to flush "
                    "the entire database. Set a prefix or call flushdb() on "
                    "the client directly."
                )
                return
            client = await self._get_async_client()
            pattern = f"{self._prefix}{KEY_SEPARATOR}*"
            all_keys = await scan_keys_async(client, pattern, self._encoding)
            if all_keys:
                await client.delete(all_keys)
        except Exception:
            logger.exception("Error clearing Valkey storage")

    def child(self, name: str | None) -> "Storage":
        """Create a child storage instance with an extended prefix.

        Note: Each child lazily creates its own sync and async connections
        on first use. Creating many children is cheap, but each one that
        performs I/O will hold its own connection pair.
        """
        if name is None:
            return self
        child_prefix = (
            f"{self._prefix}{KEY_SEPARATOR}{name}" if self._prefix else name
        )
        return ValkeyStorage(
            host=self._host,
            port=self._port,
            db=self._db,
            prefix=child_prefix,
            encoding=self._encoding,
            request_timeout=self._request_timeout,
        )

    def keys(self) -> list[str]:
        """Return all keys under the current prefix."""
        prefix_pattern = (
            f"{self._prefix}{KEY_SEPARATOR}*" if self._prefix else "*"
        )

        all_keys = scan_keys_sync(self._get_sync_client(), prefix_pattern, self._encoding)

        return [
            self._strip_prefix(k)
            for k in all_keys
            if not k.endswith(METADATA_SUFFIX)
        ]

    async def get_creation_date(self, key: str) -> str:
        """Get the creation date for the given key."""
        try:
            client = await self._get_async_client()
            meta = await client.get(self._meta_key(key))
            if meta is None:
                return ""
            timestamp_str = (
                meta.decode(self._encoding) if isinstance(meta, bytes) else str(meta)
            )
            creation_time = datetime.fromtimestamp(
                float(timestamp_str), tz=timezone.utc
            )
            return get_timestamp_formatted_with_local_tz(creation_time)
        except Exception:
            logger.exception("Error getting creation date for key %s", key)
            return ""
