import json
import re
from openai import OpenAI
from src.models import Chunk, Entity, Relationship
from src.config import OPENCODE_ZEN_API_KEY, LLM_BASE_URL, LLM_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENCODE_ZEN_API_KEY, base_url=LLM_BASE_URL)
    return _client


_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extractor. Given a text chunk, extract all named entities and the relationships between them.

Rules:
- Entities are real-world objects, concepts, people, places, organizations, technologies.
- Relationships describe how entities connect (e.g., "works_for", "located_in", "part_of", "developed_by", "invented").
- Use simple relation types (lowercase, underscores).

Output ONLY valid JSON array with no markdown:
[
  {"entity": "EntityName", "type": "Person|Organization|Technology|Location|Concept|Field", "relationships": [{"target": "OtherEntity", "relation": "relation_type"}]}
]"""


def extract_knowledge(chunks: list[Chunk]) -> tuple[list[Entity], list[Relationship]]:
    client = _get_client()
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": chunk.text},
            ],
            max_tokens=2048,
            temperature=0.1,
        )

        raw = response.choices[0].message.content or "[]"
        extracted = _parse_extraction(raw)

        for item in extracted:
            entity_name = item.get("entity", "").strip()
            entity_type = item.get("type", "Concept")
            if not entity_name:
                continue

            entity_id = f"{chunk.id}_{entity_name.lower().replace(' ', '_')}"
            entity = Entity(
                id=entity_id,
                name=entity_name,
                type=entity_type,
                chunk_id=chunk.id,
            )
            all_entities.append(entity)

            for rel in item.get("relationships", []):
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "related_to").strip().lower()
                if not target:
                    continue

                target_id = f"{chunk.id}_{target.lower().replace(' ', '_')}"
                rel_id = f"{entity_id}_to_{target_id}"
                relationship = Relationship(
                    id=rel_id,
                    source_entity=entity_name,
                    target_entity=target,
                    relation_type=relation,
                    chunk_id=chunk.id,
                )
                all_relationships.append(relationship)

    return all_entities, all_relationships


def _parse_extraction(raw: str) -> list[dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        matches = re.findall(r'"entity"\s*:\s*"([^"]+)"', cleaned)
        return [{"entity": m, "type": "Concept", "relationships": []} for m in matches]
