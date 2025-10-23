import os
import json
from openai import OpenAI
from src.utils.config import get_config

cfg = get_config()

def generate_with_openai(query: str, retrieved: list) -> dict:
    """
    Generate a structured LLM answer using OpenAI API.
    Returns: {"answer": str, "sources": list[str]}
    """
    api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        if retrieved:
            return {
                "answer": retrieved[0]["text"],
                "sources": list({retrieved[0]["meta"].get("source", "unknown")}),
            }
        return {"answer": "No context available.", "sources": []}

    client = OpenAI(api_key=api_key)
    context_chunks = "\n\n".join(
        [
            f"Source: {doc['meta'].get('source','unknown')} (p.{doc['meta'].get('page','?')})\n{doc['text']}"
            for doc in retrieved
        ]
    )
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful medical assistant. "
            "Answer concisely and only based on the provided context. "
            "Always return valid JSON with fields: "
            '{"answer": str, "sources": list of unique filenames}.'
        ),
    }

    user_msg = {
        "role": "user",
        "content": f"Question: {query}\n\nContext:\n{context_chunks}",
    }
    resp = client.chat.completions.create(
        model=cfg.get("openai_llm_model", "gpt-4o-mini"),
        messages=[system_msg, user_msg],
        temperature=cfg.get("openai_temperature", 0.2),
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return {
            "answer": data.get("answer", "").strip(),
            "sources": list(set(data.get("sources", []))), 
        }
    except Exception:
        return {
            "answer": content,
            "sources": list({doc["meta"].get("source", "unknown") for doc in retrieved}),
        }
