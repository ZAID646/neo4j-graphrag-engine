from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict = {}


class Entity(BaseModel):
    id: str
    name: str
    type: str
    chunk_id: str
    properties: dict = {}


class Relationship(BaseModel):
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    chunk_id: str
    properties: dict = {}


class QueryResult(BaseModel):
    query: str
    answer: str
    vector_context: list[str]
    graph_context: list[str]
    entities: list[dict]
    relationships: list[dict]
