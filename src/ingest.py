from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from src.models import Chunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def load_documents(docs_dir: str | Path) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".txt", ".md", ".pdf"],
        filename_as_id=True,
    )

    documents = reader.load_data()
    nodes = _splitter.get_nodes_from_documents(documents)

    chunks = []
    for i, node in enumerate(nodes):
        source = node.metadata.get("file_name", "unknown")
        chunk = Chunk(
            id=f"chunk_{i}",
            text=node.text,
            source=source,
            metadata=node.metadata,
        )
        chunks.append(chunk)

    return chunks
