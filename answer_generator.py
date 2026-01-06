import os
from typing import Any, Dict, List
from openai import OpenAI

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set it in .env")
    return OpenAI(api_key=api_key)

def _dedupe(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in (urls or []):
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def generate_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ("I couldn't find anything relevant in the knowledge base. "
                "Please rephrase the question or add more details.")

    model = os.environ.get("OPENAI_CHAT_MODEL")
    if not model:
        raise RuntimeError("OPENAI_CHAT_MODEL not found. Set it in .env")

    # Build a grounded context with references
    ctx_blocks = []
    for i, h in enumerate(hits, start=1):
        refs = _dedupe(h.get("reference_urls", []))
        ref_lines = "\n".join(f"- {u}" for u in refs) if refs else "(none)"
        ctx_blocks.append(
            f"[{i}] Section: {h.get('section') or ''}\n"
            f"Matched Q: {h.get('question') or ''}\n"
            f"Answer excerpt:\n{h.get('answer_chunk') or ''}\n"
            f"Pages: {h.get('page_start')}–{h.get('page_end')}\n"
            f"References:\n{ref_lines}"
        )
    context = "\n\n---\n\n".join(ctx_blocks)

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful support assistant. "
                    "Answer using ONLY the provided context. "
                    "If the context does not contain the answer, say you don't know and ask a clarifying question. "
                    "Write in a human-friendly way (short steps/bullets). "
                    "End with a 'References' section listing the relevant URLs."
                )
            },
            {
                "role": "user",
                "content": f"User question:\n{question}\n\nContext:\n{context}"
            }
        ]
    )
    return resp.choices[0].message.content or ""
