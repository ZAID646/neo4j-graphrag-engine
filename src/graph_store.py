from neo4j import GraphDatabase
from src.models import Entity, Relationship
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE


_driver: GraphDatabase.driver | None = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


def create_entity(entity: Entity):
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        session.run(
            "MERGE (e:Entity {name: $name}) "
            "SET e.type = $type, e.chunk_id = $chunk_id",
            name=entity.name, type=entity.type, chunk_id=entity.chunk_id,
        )


def create_relationship(rel: Relationship):
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        session.run(
            "MATCH (a:Entity {name: $source}) "
            "MATCH (b:Entity {name: $target}) "
            "MERGE (a)-[r:RELATES_TO {type: $relation}]->(b) "
            "SET r.chunk_id = $chunk_id",
            source=rel.source_entity,
            target=rel.target_entity,
            relation=rel.relation_type,
            chunk_id=rel.chunk_id,
        )


def store_graph(entities: list[Entity], relationships: list[Relationship]):
    seen_names = set()
    for e in entities:
        if e.name not in seen_names:
            create_entity(e)
            seen_names.add(e.name)
    for r in relationships:
        create_relationship(r)


def get_connected_entities(entity_name: str, depth: int = 2) -> list[dict]:
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(
            f"MATCH (e:Entity {{name: $name}})-[r*1..{depth}]-(connected) "
            "RETURN connected.name AS name, connected.type AS type, "
            "reduce(s = '', rel IN r | s + type(rel) + ' -> ') AS path",
            name=entity_name,
        )
        return [dict(r) for r in result]


def extract_entity_names(query: str) -> list[str]:
    with _get_driver().session(database=NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH (e:Entity) RETURN e.name AS name LIMIT 200"
        )
        return [r["name"] for r in result]


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
