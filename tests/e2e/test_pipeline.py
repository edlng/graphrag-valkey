# Copyright (c) 2026 Valkey contributors.
# Licensed under the MIT License

"""End-to-end test: full GraphRAG pipeline (index + query) with Valkey backends.

Requirements:
- A running Valkey instance with the valkey-search module on localhost:6379.
- Ollama running on localhost:11434 with models pulled:
    ollama pull gemma4:e2b
    ollama pull nomic-embed-text
- The graphrag package installed.
"""

from pathlib import Path

import pytest

import graphrag.api as api
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag_storage import StorageConfig, create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag_vectors import IndexSchema, VectorStoreConfig, create_vector_store

VALKEY_HOST = "localhost"
VALKEY_PORT = 6379

SAMPLE_DOCUMENT = """\
Operation Dulce is a top-secret military initiative launched by the Pentagon to
locate Santa Claus's workshop at the North Pole. General Candy Cane leads the
expedition, assisted by Lieutenant Jingle Bells (logistics), Sergeant Snowball
(arctic survival specialist), and Dr. Holly Mistletoe (reindeer behavioral
scientist). After months of satellite surveillance and intercepted radio
signals on frequency 25.12 MHz, the team pinpointed an anomalous heat
signature beneath the polar ice cap. Upon breaching the underground facility,
they discovered an industrial-scale toy manufacturing operation staffed
entirely by elves operating under a strict union contract. The elves demanded
hazard pay for working with glitter and threatened to strike unless the
military agreed to leave cookies at all checkpoints. Santa himself was
unavailable for comment, reportedly on a test flight calibrating his new
GPS-guided sleigh navigation system. The Pentagon classified the findings
under codename JINGLE CLASSIFIED and issued a gag order, though leaked
documents suggest the military is now negotiating a defense contract with
the North Pole for overnight global delivery capabilities.
"""


def _build_config(input_dir: Path) -> GraphRagConfig:
    """Build a GraphRagConfig pointing at Valkey backends."""
    return GraphRagConfig.model_validate({
        "completion_models": {
            "default_completion_model": {
                "model_provider": "ollama",
                "model": "gemma4:e2b",
                "api_base": "http://localhost:11434",
                "api_key": "ollama",
            },
        },
        "embedding_models": {
            "default_embedding_model": {
                "model_provider": "ollama",
                "model": "nomic-embed-text",
                "api_base": "http://localhost:11434",
                "api_key": "ollama",
            },
        },
        "input": {
            "type": "text",
        },
        "chunking": {
            "type": "tokens",
            "size": 1200,
            "overlap": 100,
            "encoding_model": "o200k_base",
        },
        "input_storage": {
            "type": "file",
            "base_dir": str(input_dir),
        },
        "output_storage": {
            "type": "valkey",
            "host": VALKEY_HOST,
            "port": VALKEY_PORT,
            "db": 0,
            "prefix": "e2e_output",
        },
        "cache": {
            "type": "valkey",
            "host": VALKEY_HOST,
            "port": VALKEY_PORT,
            "db": 1,
            "prefix": "e2e_cache",
            "storage": None,
        },
        "vector_store": {
            "type": "valkey",
            "host": VALKEY_HOST,
            "port": VALKEY_PORT,
            "db": 2,
            "prefix": "e2e_vs",
            "vector_size": 768,
        },
        "embed_text": {
            "embedding_model_id": "default_embedding_model",
        },
    })


@pytest.fixture
def input_dir(tmp_path):
    """Create a temporary input directory with the sample document."""
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(SAMPLE_DOCUMENT)
    return tmp_path


@pytest.fixture
def config(input_dir):
    """Build the GraphRagConfig for E2E tests."""
    return _build_config(input_dir)


@pytest.mark.e2e
@pytest.mark.timeout(300)
async def test_index_and_query(config):
    """Run the full pipeline: index documents, then run global and local search."""
    # Index
    print("\n--- Indexing ---")
    results = await api.build_index(
        config=config,
        method=IndexingMethod.Standard,
        verbose=True,
    )

    errors = [r for r in results if r.error is not None]
    assert not errors, f"Indexing failed: {[f'{e.workflow}: {e.error}' for e in errors]}"
    assert len(results) > 0
    print(f"Indexing completed — {len(results)} workflow(s):")
    for r in results:
        print(f"  • {r.workflow}")

    # Verify output storage has data
    output_storage = create_storage(
        StorageConfig(
            type="valkey",
            host=VALKEY_HOST,
            port=VALKEY_PORT,
            db=0,
            prefix="e2e_output",
        )
    )
    keys = output_storage.keys()
    assert len(keys) > 0, "No keys found in output storage after indexing"
    print(f"Output storage: {len(keys)} key(s)")
    await output_storage.close()

    # Verify vector store has embeddings
    vs_config = VectorStoreConfig(
        type="valkey",
        host=VALKEY_HOST,
        port=VALKEY_PORT,
        db=2,
        prefix="e2e_vs",
    )
    print("Vector store document counts:")
    for embedding_name in ("text_unit_text", "entity_description", "community_full_content"):
        schema = IndexSchema(index_name=embedding_name, vector_size=768)
        vs = create_vector_store(vs_config, schema)
        vs.connect()
        count = vs.count()
        assert count > 0, f"No documents in vector store for {embedding_name}"
        print(f"  • {embedding_name}: {count}")
        vs.close()

    # Global search
    print("\n--- Global Search ---")
    query_storage = create_storage(
        StorageConfig(
            type="valkey",
            host=VALKEY_HOST,
            port=VALKEY_PORT,
            db=0,
            prefix="e2e_output",
        )
    )
    table_provider = create_table_provider(config.table_provider, storage=query_storage)
    reader = DataReader(table_provider)

    entities = await reader.entities()
    communities = await reader.communities()
    community_reports = await reader.community_reports()

    response, _context = await api.global_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=None,
        dynamic_community_selection=True,
        response_type="multiple paragraphs",
        query="What is Operation Dulce and who is involved?",
        verbose=True,
    )
    assert response and len(str(response)) > 0
    print(f"Response ({len(str(response))} chars):\n{str(response)[:500]}")

    # Local search
    print("\n--- Local Search ---")
    text_units = await reader.text_units()
    relationships = await reader.relationships()

    response, _context = await api.local_search(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=None,
        community_level=2,
        response_type="multiple paragraphs",
        query="Who is General Candy Cane and what did the team discover?",
        verbose=True,
    )
    assert response and len(str(response)) > 0
    print(f"Response ({len(str(response))} chars):\n{str(response)[:500]}")

    await query_storage.close()
