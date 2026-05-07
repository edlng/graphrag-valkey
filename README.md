# graphrag-valkey

Valkey storage, cache, and vector store backends for [Microsoft GraphRAG](https://github.com/microsoft/graphrag).

This package provides three pluggable backends that integrate with GraphRAG's factory pattern:

- **ValkeyStorage** — key-value storage backend (implements `graphrag_storage.Storage`)
- **ValkeyCache** — pipeline cache backend (implements `graphrag_cache.Cache`)
- **ValkeyVectorStore** — vector similarity search backend using Valkey Search / FT module (implements `graphrag_vectors.VectorStore`)

All backends use the [valkey-glide](https://github.com/valkey-io/valkey-glide) client library.

## Installation

```bash
uv add graphrag-valkey
```

Or with pip:

```bash
pip install graphrag-valkey
```

## Quick Start

Register the Valkey backends before running any GraphRAG operations:

```python
from graphrag_valkey import register_all

# Register all three backends (storage, cache, vector store)
register_all()
```

Or register individually:

```python
from graphrag_valkey.register import register_storage, register_cache, register_vector_store

register_storage()       # Registers type "valkey" for storage
register_cache()         # Registers type "valkey" for cache
register_vector_store()  # Registers type "valkey" for vector store
```

## Configuration

After registration, configure Valkey in your `settings.yaml`:

### Storage

```yaml
output_storage:
  type: valkey
  host: localhost
  port: 6379
  db: 0
  prefix: "graphrag_output"

input_storage:
  type: valkey
  host: localhost
  port: 6379
  db: 0
  prefix: "graphrag_input"
```

### Cache

```yaml
cache:
  type: valkey
  host: localhost
  port: 6379
  db: 1
  prefix: "graphrag_cache"
```

### Vector Store

```yaml
vector_store:
  type: valkey
  host: localhost
  port: 6379
  db: 2
  prefix: "graphrag_vs"
```

The vector store requires the [Valkey Search module](https://github.com/valkey-io/valkey-search) for FT.CREATE / FT.SEARCH support. You can use the [Docker image](https://hub.docker.com/r/valkey/valkey-bundle/) to quickly start the server as a container.

## Configuration Options

| Option | Default | Description |
|---|---|---|
| `host` | `localhost` | Valkey server hostname |
| `port` | `6379` | Valkey server port |
| `db` | `0` | Database number |
| `prefix` | varies | Key prefix for namespacing |
| `encoding` | `utf-8` | Text encoding |
| `request_timeout` | `1000` | Request timeout in milliseconds |

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A running Valkey instance (for integration/e2e tests)

### Setup

```bash
# Install dependencies and create the virtual environment
uv lock
uv sync --extra dev
```

### Running Tests

```bash
# Unit tests
uv run pytest tests/unit

# Integration tests (requires Valkey on localhost:6379)
uv run pytest tests/integration

# E2E tests (requires Valkey + Ollama with models pulled)
uv run pytest tests/e2e

# All tests
uv run pytest
```

To start Valkey locally with the Search module:

```bash
docker run -d --name valkey -p 6379:6379 valkey/valkey-bundle:latest
```

For E2E tests, Ollama must also be running with the required models:

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

## Architecture

This package follows GraphRAG's [external provider pattern](https://github.com/microsoft/graphrag/discussions/2211). Rather than modifying GraphRAG core, it registers implementations through the public factory APIs (`register_storage`, `register_cache`, `register_vector_store`).
