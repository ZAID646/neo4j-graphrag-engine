---
sdk: docker
app_file: app.py
---

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://zaid646-neo4j-graphrag-engine.hf.space)

# Neo4j GraphRAG Engine

A hybrid retrieval-augmented generation (RAG) system that combines vector search (Pinecone) with graph traversal (Neo4j) for globally-aware answers. Documents are ingested, chunked via LlamaIndex, enriched with LLM-extracted entities and relationships, stored in both a vector database and a graph database, and queried through a synthesizer that merges both contexts.

## Features

- **Document ingestion** — Load and chunk text/markdown files using LlamaIndex's sentence-aware splitter
- **LLM knowledge extraction** — Extract entities (people, technologies, concepts) and relationships from every chunk via DeepSeek V4
- **Dual storage** — Graph database (Neo4j Aura) for the entity-relationship network; Vector database (Pinecone) for text embeddings
- **Hybrid retrieval** — On query, perform semantic vector search for relevant chunks and Cypher graph traversal for connected entities
- **LLM synthesis** — Merge vector context and graph context into a single accurate answer
- **Local embeddings** — sentence-transformers/all-MiniLM-L6-v2 runs locally (no per-query API cost)
- **Streamlit dashboard** — Ingest documents, run queries, view source docs, and track real-time ingestion status
- **Sequential processing** — Rate-limit aware with exponential backoff retry between LLM calls

## Architecture

```
Raw Documents (.md / .txt)
  │
  v
[LlamaIndex] --> SentenceSplitter --> text chunks
  │
  ├─────────────────────────────────────────────────┐
  v                                                 v
[LLM Extractor] (DeepSeek V4)              [sentence-transformers]
  (entities + relationships                   (384d embeddings)
   per chunk)                                     │
  │                                               v
  v                                       [Pinecone Vector DB]
[Neo4j Aura Graph DB]                    (cosine similarity search)
  (Entity nodes +                                 │
   RELATES_TO edges)                              │
  │                                               │
  └───────────────────┬───────────────────────────┘
                      v
           [Hybrid Retriever]
             ├── vector top-K chunks
             └── Cypher neighbor traversal
                      │
                      v
           [LLM Synthesizer] (DeepSeek V4)
             (merge contexts --> final answer)
                      │
                      v
           [Streamlit Dashboard]
```

## Prerequisites

- Python 3.12 or later
- Neo4j Aura instance (or local Neo4j via Docker)
- Pinecone account with an API key
- OpenCode Zen API key (DeepSeek V4)

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ZAID646/neo4j-graphrag-engine.git
cd neo4j-graphrag-engine
```

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
OPENCODE_ZEN_API_KEY=your-key-here
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your-username
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=your-database
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=graphrag-embeddings
```

### 4. Run ingestion

```bash
python -c "from src.pipeline import ingest; ingest('data/sample_docs')"
```

### 5. Launch dashboard

```bash
streamlit run app.py
```

Or open the HF Space: [zaid646-neo4j-graphrag-engine.hf.space](https://zaid646-neo4j-graphrag-engine.hf.space)

## Project Structure

```
neo4j-graphrag-engine/
├── app.py                       # Streamlit entry point (HF Spaces)
├── Dockerfile                   # HF Spaces build
├── docker-compose.yml           # Local Neo4j + app
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata
├── LICENSE                      # MIT
├── .env.example                 # Environment template
├── data/
│   └── sample_docs/             # 3 sample documents for testing
├── src/
│   ├── config.py                # Environment configuration
│   ├── models.py                # Pydantic models (Chunk, Entity, Relationship, QueryResult)
│   ├── ingest.py                # LlamaIndex document loading and chunking
│   ├── embedder.py              # sentence-transformers local embedding
│   ├── extractor.py             # LLM-powered entity and relationship extraction
│   ├── graph_store.py           # Neo4j node/edge CRUD and Cypher queries
│   ├── vector_store.py          # Pinecone upsert and semantic search
│   ├── retriever.py             # Hybrid retrieval (vector + graph)
│   ├── synthesizer.py           # LLM context merging and answer generation
│   ├── pipeline.py              # ingest() and query() orchestrators
│   ├── retry.py                 # Exponential backoff retry decorator
│   └── dashboard.py             # Streamlit UI rendering
└── tests/
```

## Pipeline Stages

### Ingestion

1. **Load documents** — Reads all `.md` and `.txt` files from the specified directory
2. **Chunk** — Splits documents into overlapping chunks using LlamaIndex's `SentenceSplitter`
3. **Extract knowledge** — For each chunk, calls DeepSeek V4 to extract named entities and their relationships
4. **Store graph** — Writes entities as Neo4j nodes and relationships as edges
5. **Generate embeddings** — Encodes each chunk as a 384-dimensional vector using sentence-transformers
6. **Upsert vectors** — Stores embeddings in Pinecone with chunk metadata

### Query

1. **Embed query** — Encodes the user's question into a 384d vector
2. **Vector search** — Retrieves top-K similar chunks from Pinecone
3. **Graph traversal** — Identifies entities mentioned in the query and runs a Cypher traversal to find connected nodes
4. **Synthesize** — Merges both contexts into a single prompt and calls DeepSeek V4 for the final answer

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCODE_ZEN_API_KEY` | (required) | API key for LLM (extraction + synthesis) |
| `LLM_BASE_URL` | `https://opencode.ai/zen/v1` | LLM API base URL |
| `LLM_MODEL` | `deepseek-v4-flash-free` | LLM model identifier |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `PINECONE_API_KEY` | (required) | Pinecone API key |
| `PINECONE_INDEX_NAME` | `graphrag-embeddings` | Pinecone index name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |

## Local Development with Docker Compose

For local testing with a Neo4j instance:

```bash
docker compose up -d
```

This starts Neo4j on port 7687 with APOC plugins enabled. Update your `.env` to use `bolt://localhost:7687` with user `neo4j` and password `password`.

## Tech Stack

- **Python 3.12+** — Core runtime
- **LlamaIndex** — Document loading and intelligent chunking
- **OpenAI Python SDK** — LLM calls (DeepSeek V4 via OpenCode Zen)
- **Neo4j Python Driver** — Graph database operations
- **Pinecone Python Client** — Vector database operations
- **sentence-transformers** — Local embedding model (all-MiniLM-L6-v2)
- **Streamlit** — Interactive dashboard
- **Pydantic** — Data validation

## License

MIT License. See [LICENSE](LICENSE) for details.
