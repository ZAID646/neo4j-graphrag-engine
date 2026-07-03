from openai import OpenAI
from src.config import OPENCODE_ZEN_API_KEY, LLM_BASE_URL, LLM_MODEL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENCODE_ZEN_API_KEY, base_url=LLM_BASE_URL)
    return _client


_SYNTHESIS_SYSTEM_PROMPT = """You are a knowledge synthesis assistant. Given a user query, text context (from vector search), and graph context (entities and relationships), produce a comprehensive, accurate answer.

Rules:
- Use both contexts to form your answer.
- Cite specific entities and facts from the graph context when relevant.
- If contexts conflict, note the discrepancy.
- Keep the answer concise but thorough."""


def synthesize(query: str, vector_context: list[str], graph_context: list[str]) -> str:
    client = _get_client()

    text_section = "\n\n".join(vector_context[:5]) if vector_context else "No text context available."
    graph_section = "\n".join(graph_context[:20]) if graph_context else "No graph context available."

    user_message = (
        f"Query: {query}\n\n"
        f"Text context from vector search:\n{text_section}\n\n"
        f"Graph context (connected entities):\n{graph_section}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    return response.choices[0].message.content or ""
