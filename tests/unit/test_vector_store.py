# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""Unit tests for ValkeyVectorStore — filter compilation, key logic, result parsing."""

import numpy as np
import pytest
from graphrag_valkey.vector_store import ValkeyVectorStore, _escape_tag
from graphrag_vectors.filtering import F


@pytest.mark.parametrize(
    "raw,expected",
    [
        # No escaping needed
        ("hello", "hello"),
        ("", ""),
        # Single special characters
        ("hello world", "hello\\ world"),
        ("user@domain", "user\\@domain"),
        ("a,b.c", "a\\,b\\.c"),
        ("(test)", "\\(test\\)"),
        ("{val}", "\\{val\\}"),
        ("what?", "what\\?"),
        ("a\\b", "a\\\\b"),
        ("[0]", "\\[0\\]"),
        ("a|b", "a\\|b"),
        ('say "hi"', 'say\\ \\"hi\\"'),
        ("it's", "it\\'s"),
        ("key:value", "key\\:value"),
        ("a;b", "a\\;b"),
        ("wow!", "wow\\!"),
        ("#tag", "\\#tag"),
        ("$100", "\\$100"),
        ("50%", "50\\%"),
        ("a^b", "a\\^b"),
        ("a&b", "a\\&b"),
        ("a*b", "a\\*b"),
        ("a+=1", "a\\+\\=1"),
        ("~approx", "\\~approx"),
        ("path/to", "path\\/to"),
        ("<html>", "\\<html\\>"),
        ("a-b", "a\\-b"),
        ("a,b,c", "a\\,b\\,c"),
        # Mixed normal and special
        ("hello world (v2.1)", "hello\\ world\\ \\(v2\\.1\\)"),
    ],
)
def test_escape_tag(raw: str, expected: str) -> None:
    """_escape_tag escapes every FT.SEARCH special character."""
    assert _escape_tag(raw) == expected


def test_escape_tag_all_specials() -> None:
    """Every character in the FT.SEARCH special set is escaped exactly once."""
    specials = """,./<>{}[]\\"|':;!@#$%^&*()-+=~? """
    result = _escape_tag(specials)
    assert len(result) == len(specials) * 2
    for i, ch in enumerate(specials):
        assert result[i * 2] == "\\"
        assert result[i * 2 + 1] == ch


class TestKeyGeneration:
    """Tests for document key and index name generation."""

    def test_doc_key(self):
        store = ValkeyVectorStore(
            prefix="vs", index_name="idx1", vector_size=3
        )
        assert store._doc_key("doc123") == "vs:idx1:doc123"

    def test_index_ft_name(self):
        store = ValkeyVectorStore(
            prefix="vs", index_name="my_index", vector_size=3
        )
        assert store._index_ft_name() == "idx:vs:my_index"

    def test_key_prefix_for_index(self):
        store = ValkeyVectorStore(
            prefix="vs", index_name="idx1", vector_size=3
        )
        assert store._key_prefix_for_index() == "vs:idx1:"


class TestFilterCompilation:
    """Tests for _compile_filter and _compile_condition."""

    def _store(self) -> ValkeyVectorStore:
        return ValkeyVectorStore(
            prefix="test", index_name="idx", vector_size=3,
            fields={"os": "str", "priority": "int", "category": "str"},
        )

    def test_eq_string(self):
        store = self._store()
        result = store._compile_filter(F.os == "linux")
        assert result == "@os:{linux}"

    def test_eq_numeric(self):
        store = self._store()
        result = store._compile_filter(F.priority == 5)
        assert result == "@priority:[5 5]"

    def test_ne_string(self):
        store = self._store()
        result = store._compile_filter(F.os != "windows")
        assert result == "-@os:{windows}"

    def test_ne_numeric(self):
        store = self._store()
        result = store._compile_filter(F.priority != 3)
        assert result == "-@priority:[3 3]"

    def test_gt(self):
        store = self._store()
        result = store._compile_filter(F.priority > 2)
        assert result == "@priority:[(2 +inf]"

    def test_gte(self):
        store = self._store()
        result = store._compile_filter(F.priority >= 2)
        assert result == "@priority:[2 +inf]"

    def test_lt(self):
        store = self._store()
        result = store._compile_filter(F.priority < 5)
        assert result == "@priority:[-inf (5]"

    def test_lte(self):
        store = self._store()
        result = store._compile_filter(F.priority <= 5)
        assert result == "@priority:[-inf 5]"

    def test_in(self):
        store = self._store()
        result = store._compile_filter(F.os.in_(["linux", "macos"]))
        assert result == "@os:{linux|macos}"

    def test_not_in(self):
        store = self._store()
        result = store._compile_filter(F.os.not_in(["windows"]))
        assert result == "-@os:{windows}"

    def test_contains_raises(self):
        store = self._store()
        with pytest.raises(ValueError, match="not supported"):
            store._compile_filter(F.os.contains("inu"))

    def test_startswith(self):
        store = self._store()
        result = store._compile_filter(F.os.startswith("lin"))
        assert result == "@os:{lin*}"

    def test_and_filter(self):
        store = self._store()
        result = store._compile_filter(
            (F.os == "linux") & (F.priority > 1)
        )
        assert "(@os:{linux})" in result
        assert "(@priority:[(1 +inf])" in result

    def test_or_filter(self):
        store = self._store()
        result = store._compile_filter(
            (F.os == "linux") | (F.os == "macos")
        )
        assert result == "(@os:{linux} | @os:{macos})"

    def test_not_filter(self):
        store = self._store()
        result = store._compile_filter(~(F.os == "windows"))
        assert result == "-(@os:{windows})"

    def test_special_chars_in_value_escaped(self):
        store = self._store()
        result = store._compile_filter(F.os == "my os (v2)")
        assert result == "@os:{my\\ os\\ \\(v2\\)}"


class TestParseSearchResults:
    """Tests for _parse_search_results."""

    def _store(self) -> ValkeyVectorStore:
        return ValkeyVectorStore(
            prefix="test", index_name="idx", vector_size=3,
            fields={"title": "str"},
        )

    def test_empty_result(self):
        store = self._store()
        assert store._parse_search_results(None, None, True) == []
        assert store._parse_search_results([], None, True) == []

    def test_zero_total(self):
        store = self._store()
        assert store._parse_search_results([0, {}], None, True) == []

    def test_valid_result(self):
        store = self._store()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        raw = [
            1,
            {
                b"test:idx:doc1": {
                    b"id": b"doc1",
                    b"vector": vec,
                    b"__score": b"0.25",
                    b"title": b"Test Doc",
                    b"create_date": b"2026-01-01",
                    b"update_date": b"",
                }
            },
        ]
        results = store._parse_search_results(raw, None, True)
        assert len(results) == 1
        assert results[0].document.id == "doc1"
        assert results[0].score == pytest.approx(0.75)
        assert results[0].document.vector is not None
        assert len(results[0].document.vector) == 3
        assert results[0].document.data["title"] == "Test Doc"

    def test_include_vectors_false(self):
        store = self._store()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        raw = [
            1,
            {
                b"test:idx:doc1": {
                    b"id": b"doc1",
                    b"vector": vec,
                    b"__score": b"0.1",
                    b"create_date": b"",
                    b"update_date": b"",
                }
            },
        ]
        results = store._parse_search_results(raw, None, False)
        assert results[0].document.vector is None

    def test_select_limits_fields(self):
        store = self._store()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        raw = [
            1,
            {
                b"test:idx:doc1": {
                    b"id": b"doc1",
                    b"vector": vec,
                    b"__score": b"0.0",
                    b"title": b"Hello",
                    b"create_date": b"",
                    b"update_date": b"",
                }
            },
        ]
        # Select only "title" — should include it
        results = store._parse_search_results(raw, ["title"], True)
        assert "title" in results[0].document.data

        # Select a field that doesn't exist — data should be empty
        results = store._parse_search_results(raw, ["nonexistent"], True)
        assert results[0].document.data == {}


class TestBytesDict:
    """Tests for _bytes_dict_to_str."""

    def test_decodes_bytes_keys_and_values(self):
        store = ValkeyVectorStore(
            prefix="t", index_name="i", vector_size=3
        )
        d = {b"name": b"hello", b"count": b"42"}
        result = store._bytes_dict_to_str(d)
        assert result == {"name": "hello", "count": "42"}

    def test_preserves_vector_bytes(self):
        store = ValkeyVectorStore(
            prefix="t", index_name="i", vector_size=3
        )
        vec = b"\x00\x01\x02\x03"
        d = {b"vector": vec, b"id": b"doc1"}
        result = store._bytes_dict_to_str(d)
        # vector field should remain as bytes
        assert result["vector"] == vec
        assert result["id"] == "doc1"


# ---------------------------------------------------------------------------
# Helpers for I/O tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

from graphrag_vectors.vector_store import VectorStoreDocument


def _make_store(
    prefix: str = "vs",
    index_name: str = "idx",
    vector_size: int = 3,
    fields: dict | None = None,
) -> tuple[ValkeyVectorStore, MagicMock]:
    """Return a ValkeyVectorStore with a pre-injected sync mock client."""
    store = ValkeyVectorStore(
        prefix=prefix,
        index_name=index_name,
        vector_size=vector_size,
        fields=fields or {},
    )
    mock_client = MagicMock()
    store._client = mock_client
    return store, mock_client


# ---------------------------------------------------------------------------
# create_index()
# ---------------------------------------------------------------------------

class TestCreateIndex:
    def test_creates_index_when_not_exists(self):
        store, client = _make_store()
        # ft.info raises → index doesn't exist → ft.create is called
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.info.side_effect = Exception("no index")
            store.create_index()
            mock_ft.create.assert_called_once()

    def test_skips_creation_when_index_exists(self):
        store, client = _make_store()
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.info.return_value = {}  # index exists
            store.create_index()
            mock_ft.create.assert_not_called()

    def test_includes_extra_fields_in_schema(self):
        store, client = _make_store(fields={"title": "str", "score": "int"})
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.info.side_effect = Exception("no index")
            store.create_index()
            # ft.create(client, index_name, schema, options)
            _, _index_name, schema_arg, _options = mock_ft.create.call_args.args
            field_names = [f.name for f in schema_arg]
            assert "title" in field_names
            assert "score" in field_names


# ---------------------------------------------------------------------------
# load_documents()
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    def test_stores_document_as_hash(self):
        store, client = _make_store()
        doc = VectorStoreDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            data={},
        )
        store.load_documents([doc])
        client.hset.assert_called_once()
        key = client.hset.call_args.args[0]
        assert key == "vs:idx:doc1"

    def test_skips_document_without_vector(self):
        store, client = _make_store()
        doc = VectorStoreDocument(id="doc1", vector=None, data={})
        store.load_documents([doc])
        client.hset.assert_not_called()

    def test_stores_extra_fields(self):
        store, client = _make_store(fields={"title": "str"})
        doc = VectorStoreDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            data={"title": "Hello"},
        )
        store.load_documents([doc])
        field_map = client.hset.call_args.args[1]
        assert "title" in field_map
        assert field_map["title"] == "Hello"

    def test_vector_stored_as_float32_bytes(self):
        store, client = _make_store()
        doc = VectorStoreDocument(id="d", vector=[1.0, 2.0, 3.0], data={})
        store.load_documents([doc])
        field_map = client.hset.call_args.args[1]
        vec_bytes = field_map["vector"]
        restored = np.frombuffer(vec_bytes, dtype=np.float32).tolist()
        assert restored == pytest.approx([1.0, 2.0, 3.0])

    def test_silently_handles_hset_exception(self):
        store, client = _make_store()
        client.hset.side_effect = RuntimeError("write error")
        doc = VectorStoreDocument(id="d", vector=[0.1, 0.2, 0.3], data={})
        store.load_documents([doc])  # should not raise


# ---------------------------------------------------------------------------
# similarity_search_by_vector()
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def _raw_result(self, doc_id: str = "doc1", score: str = "0.1") -> list:
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        return [
            1,
            {
                f"vs:idx:{doc_id}".encode(): {
                    b"id": doc_id.encode(),
                    b"vector": vec,
                    b"__score": score.encode(),
                    b"create_date": b"",
                    b"update_date": b"",
                }
            },
        ]

    def test_returns_results(self):
        store, client = _make_store()
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.search.return_value = self._raw_result()
            results = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=5)
        assert len(results) == 1
        assert results[0].document.id == "doc1"
        assert results[0].score == pytest.approx(0.9)

    def test_returns_empty_on_exception(self):
        store, client = _make_store()
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.search.side_effect = RuntimeError("search failed")
            results = store.similarity_search_by_vector([0.1, 0.2, 0.3])
        assert results == []

    def test_passes_filter_to_query(self):
        from graphrag_vectors.filtering import F
        store, client = _make_store(fields={"os": "str"})
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.search.return_value = [0, {}]
            store.similarity_search_by_vector(
                [0.1, 0.2, 0.3], filters=F.os == "linux"
            )
        query_arg = mock_ft.search.call_args.args[2]
        assert "@os:{linux}" in query_arg


# ---------------------------------------------------------------------------
# search_by_id()
# ---------------------------------------------------------------------------

class TestSearchById:
    def test_returns_document(self):
        store, client = _make_store()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        client.hgetall.return_value = {
            b"id": b"doc1",
            b"vector": vec,
            b"create_date": b"2026-01-01",
            b"update_date": b"",
        }
        doc = store.search_by_id("doc1")
        assert doc.id == "doc1"
        assert doc.vector is not None
        client.hgetall.assert_called_once_with("vs:idx:doc1")

    def test_raises_index_error_when_not_found(self):
        store, client = _make_store()
        client.hgetall.return_value = {}
        with pytest.raises(IndexError):
            store.search_by_id("missing")

    def test_excludes_vector_when_include_vectors_false(self):
        store, client = _make_store()
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        client.hgetall.return_value = {
            b"id": b"doc1",
            b"vector": vec,
            b"create_date": b"",
            b"update_date": b"",
        }
        doc = store.search_by_id("doc1", include_vectors=False)
        assert doc.vector is None


# ---------------------------------------------------------------------------
# count()
# ---------------------------------------------------------------------------

class TestCount:
    def test_returns_num_docs(self):
        store, client = _make_store()
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.info.return_value = {b"num_docs": 42}
            assert store.count() == 42

    def test_returns_zero_when_index_missing(self):
        store, client = _make_store()
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.info.side_effect = Exception("no index")
            assert store.count() == 0


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------

class TestRemove:
    def test_deletes_each_document(self):
        store, client = _make_store()
        store.remove(["doc1", "doc2"])
        assert client.delete.call_count == 2
        deleted_keys = [call.args[0][0] for call in client.delete.call_args_list]
        assert "vs:idx:doc1" in deleted_keys
        assert "vs:idx:doc2" in deleted_keys

    def test_silently_handles_exception(self):
        store, client = _make_store()
        client.delete.side_effect = RuntimeError("gone")
        store.remove(["doc1"])  # should not raise


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_updates_fields(self):
        store, client = _make_store(fields={"title": "str"})
        doc = VectorStoreDocument(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            data={"title": "Updated"},
        )
        store.update(doc)
        client.hset.assert_called_once()
        key, field_map = client.hset.call_args.args
        assert key == "vs:idx:doc1"
        assert "title" in field_map
        assert field_map["title"] == "Updated"

    def test_updates_vector_when_provided(self):
        store, client = _make_store()
        doc = VectorStoreDocument(id="doc1", vector=[1.0, 0.0, 0.0], data={})
        store.update(doc)
        field_map = client.hset.call_args.args[1]
        assert "vector" in field_map
        restored = np.frombuffer(field_map["vector"], dtype=np.float32).tolist()
        assert restored == pytest.approx([1.0, 0.0, 0.0])

    def test_silently_handles_exception(self):
        store, client = _make_store()
        client.hset.side_effect = RuntimeError("write error")
        doc = VectorStoreDocument(id="doc1", vector=[0.1, 0.2, 0.3], data={})
        store.update(doc)  # should not raise


# ---------------------------------------------------------------------------
# close() / drop_index()
# ---------------------------------------------------------------------------

class TestVectorStoreClose:
    def test_closes_client(self):
        store, client = _make_store()
        store.close()
        client.close.assert_called_once()
        assert store._client is None

    def test_close_is_idempotent(self):
        store, client = _make_store()
        store.close()
        store.close()  # second call should not raise
        client.close.assert_called_once()


class TestDropIndex:
    def test_drops_index_and_deletes_keys(self):
        store, client = _make_store()
        client.scan.return_value = (b"0", [b"vs:idx:doc1"])
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            store.drop_index()
            client.delete.assert_called_once_with(["vs:idx:doc1"])
            mock_ft.dropindex.assert_called_once_with(client, "idx:vs:idx")

    def test_tolerates_missing_index_on_drop(self):
        store, client = _make_store()
        client.scan.return_value = (b"0", [])
        with patch("graphrag_valkey.vector_store.ft") as mock_ft:
            mock_ft.dropindex.side_effect = Exception("no index")
            store.drop_index()  # should not raise
