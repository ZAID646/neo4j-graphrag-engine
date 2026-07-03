import os
from dotenv import load_dotenv

load_dotenv()

OPENCODE_ZEN_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash-free")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "graphrag-embeddings")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
