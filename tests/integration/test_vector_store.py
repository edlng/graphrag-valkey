# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Integration tests for ValkeyVectorStore.

Requires a running Valkey instance with the valkey-search module on localhost:6379.
"""

import uuid

import numpy as np
import pytest
from graphrag_valkey import ValkeyVectorStore
from graphrag_vectors import VectorStoreDocument
from graphrag_vectors.filtering import F


def _unique_prefix() -> str:
    """Generate a unique prefix to isolate test runs."""
    return f"test_vs_{uuid.uuid4().hex[:8]}"


def _cleanup_store(store: ValkeyVectorStore) -> None:
    """Remove all documents and drop the FT index for cleanup."""
    store.drop_index()
    store.close()


class TestValkeyVectorStore:
    """Tests for ValkeyVectorStore."""

    @pytest.fixture
    def prefix(self):
        """Unique prefix per test."""
        return _unique_prefix()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
            VectorStoreDocument(id="2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
            VectorStoreDocument(id="3", vector=[0.3, 0.4, 0.5, 0.6, 0.7]),
        ]

    @pytest.fixture
    def sample_documents_with_metadata(self):
        """Create sample documents with metadata fields for testing."""
        return [
            VectorStoreDocument(
                id="1",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                data={"os": "windows", "category": "bug", "priority": 1},
            ),
            VectorStoreDocument(
                id="2",
                vector=[0.2, 0.3, 0.4, 0.5, 0.6],
                data={"os": "linux", "category": "feature", "priority": 2},
            ),
            VectorStoreDocument(
                id="3",
                vector=[0.3, 0.4, 0.5, 0.6, 0.7],
                data={"os": "windows", "category": "feature", "priority": 3},
            ),
        ]

    @pytest.fixture
    def store(self, prefix):
        """Create a basic ValkeyVectorStore."""
        s = ValkeyVectorStore(
            host="localhost",
            port=6379,
            db=0,
            prefix=prefix,
            index_name="test_index",
            vector_size=5,
        )
        s.connect()
        s.create_index()
        yield s
        _cleanup_store(s)

    @pytest.fixture
    def store_with_fields(self, prefix):
        """Create a ValkeyVectorStore with metadata fields configured."""
        s = ValkeyVectorStore(
            host="localhost",
            port=6379,
            db=0,
            prefix=prefix,
            index_name="test_fields",
            vector_size=5,
            fields={"os": "str", "category": "str", "priority": "int"},
        )
        s.connect()
        s.create_index()
        yield s
        _cleanup_store(s)

    def test_create_index_and_count_empty(self, store):
        assert store.count() == 0

    def test_load_documents(self, store, sample_documents):
        store.load_documents(sample_documents)
        assert store.count() == 3

    def test_insert_single(self, store):
        doc = VectorStoreDocument(id="single", vector=[0.5, 0.5, 0.5, 0.5, 0.5])
        store.insert(doc)
        assert store.count() == 1

    def test_search_by_id(self, store, sample_documents):
        store.load_documents(sample_documents)
        doc = store.search_by_id("1")
        assert doc.id == "1"
        assert doc.vector is not None
        assert np.allclose(doc.vector, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)

    def test_search_by_id_not_found(self, store):
        with pytest.raises(IndexError):
            store.search_by_id("nonexistent")

    def test_search_by_id_with_fields(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        doc = store_with_fields.search_by_id("1")
        assert doc.id == "1"
        assert doc.data["os"] == "windows"
        assert doc.data["category"] == "bug"
        assert doc.data["priority"] == 1

    def test_similarity_search_by_vector(self, store, sample_documents):
        store.load_documents(sample_documents)
        results = store.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4, 0.5], k=3)
        assert len(results) == 3
        assert results[0].document.id == "1"
        assert results[0].score >= results[1].score

    def test_similarity_search_k_limit(self, store, sample_documents):
        store.load_documents(sample_documents)
        results = store.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4, 0.5], k=1)
        assert len(results) == 1
        assert results[0].document.id == "1"

    def test_similarity_search_by_text(self, store, sample_documents):
        store.load_documents(sample_documents)

        def mock_embedder(text: str) -> list[float]:
            return [0.1, 0.2, 0.3, 0.4, 0.5]

        results = store.similarity_search_by_text("test", mock_embedder, k=2)
        assert len(results) == 2
        assert results[0].document.id == "1"
        assert results[1].document.id == "2"

    def test_similarity_search_include_vectors_false(self, store, sample_documents):
        store.load_documents(sample_documents)
        results = store.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=1, include_vectors=False
        )
        assert results[0].document.vector is None

    def test_fields_returned_in_search(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=1
        )
        assert results[0].document.data["os"] == "windows"
        assert results[0].document.data["category"] == "bug"
        assert results[0].document.data["priority"] == 1

    def test_select_limits_fields(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=1, select=["os"]
        )
        data = results[0].document.data
        assert "os" in data
        assert "category" not in data
        assert "priority" not in data

    def test_select_on_search_by_id(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        doc = store_with_fields.search_by_id("1", select=["os"])
        assert "os" in doc.data
        assert "category" not in doc.data

    def test_remove(self, store, sample_documents):
        store.load_documents(sample_documents)
        assert store.count() == 3
        store.remove(["1", "2"])
        assert store.count() == 1
        with pytest.raises(IndexError):
            store.search_by_id("1")
        doc = store.search_by_id("3")
        assert doc.id == "3"

    def test_update(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        store_with_fields.update(
            VectorStoreDocument(
                id="1",
                vector=None,
                data={"os": "macos", "category": "bug", "priority": 1},
            )
        )
        doc = store_with_fields.search_by_id("1")
        assert doc.data["os"] == "macos"

    def test_update_sets_update_date(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        store_with_fields.update(
            VectorStoreDocument(id="1", vector=None, data={"os": "macos"})
        )
        doc = store_with_fields.search_by_id("1")
        assert doc.update_date is not None
        assert doc.update_date != ""

    def test_create_date_auto_set(self, store):
        store.insert(VectorStoreDocument(id="auto_date", vector=[0.1, 0.2, 0.3, 0.4, 0.5]))
        doc = store.search_by_id("auto_date")
        assert doc.create_date is not None
        assert doc.create_date != ""

    def test_filter_eq(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.os == "linux"
        )
        assert len(results) == 1
        assert results[0].document.id == "2"

    def test_filter_ne(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.os != "linux"
        )
        assert len(results) == 2
        ids = {r.document.id for r in results}
        assert ids == {"1", "3"}

    def test_filter_gt(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.priority > 1
        )
        assert len(results) == 2

    def test_filter_gte(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.priority >= 2
        )
        assert len(results) == 2

    def test_filter_lt(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.priority < 3
        )
        assert len(results) == 2

    def test_filter_lte(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.priority <= 1
        )
        assert len(results) == 1

    def test_filter_and(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            k=10,
            filters=(F.os == "windows") & (F.category == "feature"),
        )
        assert len(results) == 1
        assert results[0].document.id == "3"

    def test_filter_or(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            k=10,
            filters=(F.os == "linux") | (F.category == "bug"),
        )
        assert len(results) == 2
        ids = {r.document.id for r in results}
        assert ids == {"1", "2"}

    def test_filter_not(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=~(F.os == "windows")
        )
        assert len(results) == 1
        assert results[0].document.id == "2"

    def test_filter_in(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            k=10,
            filters=F.os.in_(["windows", "macos"]),
        )
        assert len(results) == 2
        ids = {r.document.id for r in results}
        assert ids == {"1", "3"}

    def test_filter_combined_with_search(self, store_with_fields, sample_documents_with_metadata):
        store_with_fields.load_documents(sample_documents_with_metadata)
        results = store_with_fields.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5], k=10, filters=F.category == "feature"
        )
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_search_by_id_include_vectors_false(self, store, sample_documents):
        """search_by_id with include_vectors=False must omit the vector."""
        store.load_documents(sample_documents)
        doc = store.search_by_id("1", include_vectors=False)
        assert doc.id == "1"
        assert doc.vector is None

    def test_create_date_components(self, store):
        """Exploded create_date component fields are persisted and retrievable."""
        store.insert(
            VectorStoreDocument(
                id="dated",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                create_date="2024-03-15T14:30:00",
            )
        )
        doc = store.search_by_id("dated")
        assert doc.data["create_date_year"] == 2024
        assert doc.data["create_date_month"] == 3
        assert doc.data["create_date_month_name"] == "March"
        assert doc.data["create_date_day"] == 15
        assert doc.data["create_date_day_of_week"] == "Friday"
        assert doc.data["create_date_hour"] == 14
        assert doc.data["create_date_quarter"] == 1

    def test_filter_by_timestamp_component(self, store):
        """Filtering on an exploded timestamp component works via FT.SEARCH."""
        store.insert(
            VectorStoreDocument(
                id="dec",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                create_date="2024-12-25T10:00:00",
            )
        )
        store.insert(
            VectorStoreDocument(
                id="mar",
                vector=[0.2, 0.3, 0.4, 0.5, 0.6],
                create_date="2024-03-15T10:00:00",
            )
        )

        results = store.similarity_search_by_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            k=10,
            filters=F.create_date_month == 12,
        )
        assert len(results) == 1
        assert results[0].document.id == "dec"

    def test_user_defined_date_field_exploded(self, prefix):
        """A field declared as type 'date' is exploded into filterable components."""
        s = ValkeyVectorStore(
            host="localhost",
            port=6379,
            db=0,
            prefix=prefix,
            index_name="date_field_test",
            vector_size=5,
            fields={"published_at": "date", "category": "str"},
        )
        s.connect()
        s.create_index()
        try:
            s.insert(
                VectorStoreDocument(
                    id="pub1",
                    vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                    data={"published_at": "2024-07-04T12:00:00", "category": "news"},
                )
            )

            doc = s.search_by_id("pub1")
            assert doc.data["published_at_year"] == 2024
            assert doc.data["published_at_month"] == 7
            assert doc.data["published_at_month_name"] == "July"
            assert doc.data["published_at_quarter"] == 3

            results = s.similarity_search_by_vector(
                [0.1, 0.2, 0.3, 0.4, 0.5],
                k=10,
                filters=F.published_at_month == 7,
            )
            assert len(results) == 1
            assert results[0].document.id == "pub1"
        finally:
            _cleanup_store(s)

    def test_vector_store_customization(self, prefix):
        """Custom id_field and vector_field names flow through the full pipeline."""
        s = ValkeyVectorStore(
            host="localhost",
            port=6379,
            db=0,
            prefix=prefix,
            index_name="custom_fields",
            id_field="id_custom",
            vector_field="vector_custom",
            vector_size=5,
        )
        s.connect()
        s.create_index()
        try:
            docs = [
                VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3, 0.4, 0.5]),
                VectorStoreDocument(id="2", vector=[0.2, 0.3, 0.4, 0.5, 0.6]),
            ]
            s.load_documents(docs)

            doc = s.search_by_id("1")
            assert doc.id == "1"
            assert doc.vector is not None
            assert np.allclose(doc.vector, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)

            results = s.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4, 0.5], k=2)
            assert 1 <= len(results) <= 2
            assert isinstance(results[0].score, float)

            with pytest.raises(IndexError):
                s.search_by_id("nonexistent")
        finally:
            _cleanup_store(s)
