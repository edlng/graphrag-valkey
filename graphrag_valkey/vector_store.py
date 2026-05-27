# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Valkey Vector Store implementation for GraphRAG.

Uses Valkey hash keys to store documents and the FT.SEARCH module
(valkey-search) for vector similarity search.
"""

import json
import logging
import re
import threading
from contextlib import suppress
from typing import Any

import numpy as np
from glide_sync import (
    DataType,
    DistanceMetricType,
    Field,
    FtCreateOptions,
    FtSearchLimit,
    FtSearchOptions,
    GlideClient,
    NumericField,
    TagField,
    TextField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesHnsw,
    VectorType,
    ft,
)
from graphrag_vectors.filtering import (
    AndExpr,
    Condition,
    FilterExpr,
    NotExpr,
    Operator,
    OrExpr,
)
from graphrag_vectors.vector_store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

from graphrag_valkey.client import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
    KEY_SEPARATOR,
    create_sync_client,
    scan_keys_sync,
)

logger = logging.getLogger(__name__)


class ValkeyVectorStore(VectorStore):
    """Valkey-backed vector store using FT.CREATE / FT.SEARCH."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        db: int = DEFAULT_DB,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        prefix: str = "graphrag_vs",
        **kwargs: Any,
    ) -> None:
        """Create a Valkey vector store instance."""
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._db = db
        self._request_timeout = request_timeout
        self._prefix = prefix
        self._client: GlideClient | None = None
        self._client_lock = threading.Lock()
        self._encoding = "utf-8"
        logger.info(
            "Creating Valkey vector store at %s:%d (db=%d, index=%s)",
            self._host,
            self._port,
            self._db,
            self.index_name,
        )

    def _get_client(self) -> GlideClient:
        """Lazily initialize and return the synchronous Valkey client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = create_sync_client(
                        self._host, self._port, self._db, self._request_timeout
                    )
        return self._client

    def close(self) -> None:
        """Close the underlying Valkey client connection."""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def drop_index(self) -> None:
        """Drop the FT.SEARCH index and remove all indexed documents.

        This removes the index definition and deletes all hash keys under
        the index prefix. Useful for teardown in tests or full resets.
        """
        client = self._get_client()
        ft_name = self._index_ft_name()
        prefix = self._key_prefix_for_index()

        # Delete all hash keys under the index prefix
        all_keys = scan_keys_sync(client, f"{prefix}*", self._encoding)
        if all_keys:
            client.delete(all_keys)

        # Drop the FT index
        try:
            ft.dropindex(client, ft_name)
        except Exception:  # noqa: BLE001
            logger.debug("Index %s did not exist or could not be dropped", ft_name)

    def _doc_key(self, doc_id: str | int) -> str:
        """Return the Valkey hash key for a document."""
        return f"{self._prefix}{KEY_SEPARATOR}{self.index_name}{KEY_SEPARATOR}{doc_id}"

    def _index_ft_name(self) -> str:
        """Return the FT index name."""
        return f"idx:{self._prefix}:{self.index_name}"

    def _key_prefix_for_index(self) -> str:
        """Return the key prefix that the FT index covers."""
        return f"{self._prefix}{KEY_SEPARATOR}{self.index_name}{KEY_SEPARATOR}"

    def connect(self) -> None:
        """Connect to vector storage."""
        self._get_client()

    def create_index(self) -> None:
        """Create the FT.SEARCH index."""
        client = self._get_client()
        ft_name = self._index_ft_name()
        prefix = self._key_prefix_for_index()

        # Check if index already exists
        try:
            ft.info(client, ft_name)
        except Exception:  # noqa: BLE001
            logger.debug("Index %s does not exist, creating", ft_name)
        else:
            logger.info("Index %s already exists", ft_name)
            return

        # Build schema using glide Field objects
        schema: list[Field] = []

        # ID field (TAG for exact match)
        schema.append(TagField(self.id_field))

        # Vector field (HNSW)
        schema.append(
            VectorField(
                self.vector_field,
                algorithm=VectorAlgorithm.HNSW,
                attributes=VectorFieldAttributesHnsw(
                    dimensions=self.vector_size,
                    distance_metric=DistanceMetricType.COSINE,
                    type=VectorType.FLOAT32,
                ),
            )
        )

        # Date fields
        schema.append(TextField(self.create_date_field))
        schema.append(TextField(self.update_date_field))

        # Additional fields
        for field_name, field_type in self.fields.items():
            if field_type in ("int", "float"):
                schema.append(NumericField(field_name))
            else:
                # TagField supports exact-match and tag-style filter
                # queries (@field:{value}) used by _compile_filter.
                schema.append(TagField(field_name))

        options = FtCreateOptions(data_type=DataType.HASH, prefixes=[prefix])

        try:
            ft.create(client, ft_name, schema, options)
            logger.info("Created FT index %s", ft_name)
        except Exception:
            logger.exception("Error creating FT index %s", ft_name)
            raise

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Load documents into Valkey as hash keys."""
        client = self._get_client()

        for document in documents:
            self._prepare_document(document)
            if document.vector is None:
                continue

            doc_key = self._doc_key(document.id)
            vector_bytes = np.array(
                document.vector, dtype=np.float32
            ).tobytes()

            field_value_map: dict[str | bytes, str | bytes] = {
                self.id_field: str(document.id),
                self.vector_field: vector_bytes,
                self.create_date_field: document.create_date or "",
                self.update_date_field: document.update_date or "",
            }

            if document.data:
                for field_name in self.fields:
                    if field_name in document.data:
                        val = document.data[field_name]
                        field_value_map[field_name] = (
                            json.dumps(val) if not isinstance(val, str) else val
                        )

            try:
                client.hset(doc_key, field_value_map)
            except Exception:
                logger.exception("Error loading document %s", document.id)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: FilterExpr | None = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by vector using FT.SEARCH."""
        client = self._get_client()
        ft_name = self._index_ft_name()

        query_blob = np.array(query_embedding, dtype=np.float32).tobytes()

        # Build filter expression
        filter_str = self._compile_filter(filters) if filters else "*"

        query = (
            f"({filter_str})=>"
            f"[KNN {k} @{self.vector_field} $query_vec AS __score]"
        )

        search_options = FtSearchOptions(
            params={"query_vec": query_blob},
            limit=FtSearchLimit(offset=0, count=k),
        )

        try:
            raw_result = ft.search(client, ft_name, query, search_options)
        except Exception:
            logger.exception("Error performing similarity search")
            return []

        return self._parse_search_results(raw_result, select, include_vectors)

    def _parse_search_results(
        self,
        raw_result: Any,
        select: list[str] | None,
        include_vectors: bool,
    ) -> list[VectorStoreSearchResult]:
        """Parse FT.SEARCH results into VectorStoreSearchResult list.

        The ft.search response format is:
        [total_count, {doc_key: {field: value, ...}, ...}]
        """
        results: list[VectorStoreSearchResult] = []

        if not raw_result or not isinstance(raw_result, list):
            return results

        total = raw_result[0] if raw_result else 0
        if isinstance(total, bytes):
            total = int(total)
        if total == 0:
            return results

        # Second element is a mapping of doc_key -> field_map
        if len(raw_result) < 2 or not isinstance(raw_result[1], dict):
            return results

        doc_map: dict[Any, Any] = raw_result[1]

        for field_map in doc_map.values():
            if not isinstance(field_map, dict):
                continue

            doc_dict = self._bytes_dict_to_str(field_map)

            doc_id = doc_dict.get(self.id_field, "")
            score_raw = doc_dict.get("__score", "1.0")
            score = 1.0 - float(score_raw)  # Convert distance to similarity

            vector = None
            if include_vectors and self.vector_field in doc_dict:
                vec_bytes = doc_dict[self.vector_field]
                if isinstance(vec_bytes, bytes):
                    vector = np.frombuffer(vec_bytes, dtype=np.float32).tolist()

            data = self._extract_data(doc_dict, select)

            results.append(
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=doc_id,
                        vector=vector,
                        data=data,
                        create_date=doc_dict.get(self.create_date_field),
                        update_date=doc_dict.get(self.update_date_field),
                    ),
                    score=score,
                )
            )

        return results

    def _bytes_dict_to_str(self, d: dict[Any, Any]) -> dict[str, Any]:
        """Convert a dict with bytes keys/values to string keys, decoding non-vector values.

        The field named ``self.vector_field`` is intentionally left as raw bytes
        so callers can pass it directly to ``np.frombuffer`` without an extra
        encode/decode round-trip.
        """
        result: dict[str, Any] = {}
        for key, value in d.items():
            str_key = key.decode(self._encoding) if isinstance(key, bytes) else str(key)
            if str_key == self.vector_field:
                result[str_key] = value
            elif isinstance(value, bytes):
                result[str_key] = value.decode(self._encoding)
            else:
                result[str_key] = value
        return result

    def _extract_data(
        self, doc: dict[str, Any], select: list[str] | None = None
    ) -> dict[str, Any]:
        """Extract additional field data from a document response."""
        fields_to_extract = select if select is not None else list(self.fields.keys())
        data: dict[str, Any] = {}
        for field_name in fields_to_extract:
            if field_name in doc:
                val = doc[field_name]
                # Try to deserialize JSON values
                if isinstance(val, str):
                    with suppress(json.JSONDecodeError, ValueError):
                        val = json.loads(val)
                data[field_name] = val
        return data

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Search for a document by id."""
        client = self._get_client()
        doc_key = self._doc_key(id)

        try:
            result = client.hgetall(doc_key)
        except Exception:
            logger.exception("Error searching for document %s", id)
            raise

        if not result:
            msg = f"Document with id '{id}' not found."
            raise IndexError(msg)

        doc_dict = self._bytes_dict_to_str(result)

        vector = None
        if include_vectors and self.vector_field in doc_dict:
            vec_bytes = doc_dict[self.vector_field]
            if isinstance(vec_bytes, bytes):
                vector = np.frombuffer(vec_bytes, dtype=np.float32).tolist()

        return VectorStoreDocument(
            id=doc_dict.get(self.id_field, id),
            vector=vector,
            data=self._extract_data(doc_dict, select),
            create_date=doc_dict.get(self.create_date_field),
            update_date=doc_dict.get(self.update_date_field),
        )

    def count(self) -> int:
        """Return the total number of documents in the store.

        Note: Uses FT.INFO num_docs which may be eventually consistent
        and can lag slightly behind the actual document count.
        Returns 0 if the index does not exist yet.
        """
        client = self._get_client()
        ft_name = self._index_ft_name()
        try:
            result = ft.info(client, ft_name)
            num_docs = result.get(b"num_docs", 0)
            return int(num_docs)
        except Exception:
            logger.debug("Index %s not found or error counting documents", ft_name)
        return 0

    def remove(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        client = self._get_client()
        for doc_id in ids:
            doc_key = self._doc_key(doc_id)
            try:
                client.delete([doc_key])
            except Exception:
                logger.exception("Error removing document %s", doc_id)

    def update(self, document: VectorStoreDocument) -> None:
        """Update an existing document in the store."""
        self._prepare_update(document)
        client = self._get_client()
        doc_key = self._doc_key(document.id)

        field_value_map: dict[str | bytes, str | bytes] = {
            self.update_date_field: document.update_date or "",
        }

        if document.vector is not None:
            vector_bytes = np.array(
                document.vector, dtype=np.float32
            ).tobytes()
            field_value_map[self.vector_field] = vector_bytes

        if document.data:
            for field_name in self.fields:
                if field_name in document.data:
                    val = document.data[field_name]
                    field_value_map[field_name] = (
                        json.dumps(val) if not isinstance(val, str) else val
                    )

        try:
            client.hset(doc_key, field_value_map)
        except Exception:
            logger.exception("Error updating document %s", document.id)

    def _compile_filter(self, expr: FilterExpr) -> str:
        """Compile a FilterExpr into a Valkey FT.SEARCH query filter."""
        match expr:
            case Condition():
                return self._compile_condition(expr)
            case AndExpr():
                parts = [self._compile_filter(e) for e in expr.and_]
                return " ".join(f"({p})" for p in parts)
            case OrExpr():
                parts = [self._compile_filter(e) for e in expr.or_]
                return "(" + " | ".join(parts) + ")"
            case NotExpr():
                inner = self._compile_filter(expr.not_)
                return f"-({inner})"
            case _:
                msg = f"Unsupported filter expression type: {type(expr)}"
                raise ValueError(msg)

    def _compile_condition(self, cond: Condition) -> str:
        """Compile a single Condition to Valkey FT.SEARCH syntax."""
        field = cond.field
        value = cond.value

        match cond.operator:
            case Operator.eq:
                if isinstance(value, str):
                    return f"@{field}:{{{_escape_tag(value)}}}"
                return f"@{field}:[{value} {value}]"
            case Operator.ne:
                if isinstance(value, str):
                    return f"-@{field}:{{{_escape_tag(value)}}}"
                return f"-@{field}:[{value} {value}]"
            case Operator.gt:
                return f"@{field}:[({value} +inf]"
            case Operator.gte:
                return f"@{field}:[{value} +inf]"
            case Operator.lt:
                return f"@{field}:[-inf ({value}]"
            case Operator.lte:
                return f"@{field}:[-inf {value}]"
            case Operator.in_:
                # Only works on TagField-indexed fields. Numeric fields
                # use NumericField and won't match tag-style queries.
                tags = "|".join(_escape_tag(str(v)) for v in value)
                return f"@{field}:{{{tags}}}"
            case Operator.not_in:
                tags = "|".join(_escape_tag(str(v)) for v in value)
                return f"-@{field}:{{{tags}}}"
            case Operator.contains:
                msg = (
                    "Operator 'contains' is not supported by Valkey FT.SEARCH. "
                    "Only prefix (startswith) and suffix matching are available."
                )
                raise ValueError(msg)
            case Operator.startswith:
                return f"@{field}:{{{_escape_tag(value)}*}}"
            case _:
                msg = f"Unsupported operator for Valkey FT.SEARCH: {cond.operator}"
                raise ValueError(msg)


_TAG_ESCAPE_RE = re.compile(r"""([,./<>{}\[\]\\"|':;!@#$%^&*()\-+=~? ])""")


def _escape_tag(value: str) -> str:
    """Escape special characters for Valkey FT.SEARCH tag queries.

    The full set of characters requiring escaping in tag values includes
    all punctuation that has special meaning in the FT.SEARCH query syntax.
    """
    return _TAG_ESCAPE_RE.sub(r"\\\1", value)
