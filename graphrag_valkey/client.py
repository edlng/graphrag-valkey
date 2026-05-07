# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Shared Valkey client utilities."""

import logging

from glide import GlideClient as AsyncGlideClient
from glide import GlideClientConfiguration as AsyncGlideClientConfiguration
from glide import NodeAddress as AsyncNodeAddress
from glide_sync import GlideClient, GlideClientConfiguration, NodeAddress

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6379
DEFAULT_DB = 0
DEFAULT_REQUEST_TIMEOUT = 1000
KEY_SEPARATOR = ":"
METADATA_SUFFIX = ":__meta__"


def create_sync_client(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db: int = DEFAULT_DB,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> GlideClient:
    """Create and return a connected synchronous GlideClient instance.

    Parameters
    ----------
    host : str
        Valkey server hostname.
    port : int
        Valkey server port.
    db : int
        Valkey database number.
    request_timeout : int
        Request timeout in milliseconds.

    Returns
    -------
    GlideClient
        A connected synchronous Valkey client.
    """
    config = GlideClientConfiguration(
        addresses=[NodeAddress(host, port)],
        request_timeout=request_timeout,
        database_id=db,
    )
    logger.info("Connecting (sync) to Valkey at %s:%d (db=%d)", host, port, db)
    return GlideClient.create(config)


def scan_keys_sync(
    client: GlideClient, pattern: str, encoding: str = "utf-8"
) -> list[str]:
    """Scan for keys matching a glob pattern using SCAN (sync).

    Returns all matched keys as decoded strings. Note that SCAN iterates
    over multiple round-trips, so results are a best-effort snapshot.
    """
    matched: list[str] = []
    cursor = "0"
    while True:
        result = client.scan(cursor, match=pattern)
        keys = result[1]
        matched.extend(
            k.decode(encoding) if isinstance(k, bytes) else str(k) for k in keys
        )
        cursor_bytes = result[0]
        cursor = (
            cursor_bytes.decode(encoding)
            if isinstance(cursor_bytes, bytes)
            else str(cursor_bytes)
        )
        if cursor == "0":
            break
    return matched


async def scan_keys_async(
    client: AsyncGlideClient, pattern: str, encoding: str = "utf-8"
) -> list[str]:
    """Scan for keys matching a glob pattern using SCAN (async).

    Returns all matched keys as decoded strings. Note that SCAN iterates
    over multiple round-trips, so results are a best-effort snapshot.
    """
    matched: list[str] = []
    cursor = "0"
    while True:
        result = await client.scan(cursor, match=pattern)
        keys = result[1]
        matched.extend(
            k.decode(encoding) if isinstance(k, bytes) else str(k) for k in keys
        )
        cursor_bytes = result[0]
        cursor = (
            cursor_bytes.decode(encoding)
            if isinstance(cursor_bytes, bytes)
            else str(cursor_bytes)
        )
        if cursor == "0":
            break
    return matched


async def create_async_client(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db: int = DEFAULT_DB,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> AsyncGlideClient:
    """Create and return a connected async GlideClient instance.

    Parameters
    ----------
    host : str
        Valkey server hostname.
    port : int
        Valkey server port.
    db : int
        Valkey database number.
    request_timeout : int
        Request timeout in milliseconds.

    Returns
    -------
    AsyncGlideClient
        A connected async Valkey client.
    """
    config = AsyncGlideClientConfiguration(
        addresses=[AsyncNodeAddress(host, port)],
        request_timeout=request_timeout,
        database_id=db,
    )
    logger.info("Connecting (async) to Valkey at %s:%d (db=%d)", host, port, db)
    return await AsyncGlideClient.create(config)
