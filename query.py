import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from chroma_client import get_chroma_client, get_collection
from embeddings import embed_texts, DEFAULT_EMBED_MODEL

def query_kb(
    question: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "support-kb")
    embed_model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    client = get_chroma_client(persist_dir=persist_dir)
    collection = get_collection(client, name=collection_name)

    qvec = embed_texts([question], model=embed_model)[0]

    where = None
    if filters:
        where = {k: v for k, v in filters.items() if v is not None}

    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"]
    )

    ids = (res.get("ids") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for i, _id in enumerate(ids):
        md = metadatas[i] if i < len(metadatas) else {}
        ref_str = md.get("reference_urls") or ""
        out.append({
            "id": _id,
            "distance": float(distances[i]) if i < len(distances) else None,
            "section": md.get("section"),
            "question": md.get("question"),
            "answer_chunk": md.get("answer_chunk"),
            "reference_urls": ref_str.split(" | ") if ref_str else [],
            "source": md.get("source"),
            "page_start": md.get("page_start"),
            "page_end": md.get("page_end"),
            "doc_id": md.get("doc_id"),
            "chunk_index": md.get("chunk_index"),
        })
    return out

if __name__ == "__main__":
    load_dotenv()
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do I create an Elsevier account?"
    hits = query_kb(q, top_k=5)
    for h in hits:
        print(f"\nDistance: {h['distance']}")
        print(f"Section: {h['section']}")
        print(f"Q: {h['question']}")
        print(f"A chunk: {str(h['answer_chunk'])[:500]}...")
        print(f"Refs: {h['reference_urls']}")
        print(f"Pages: {h['page_start']}–{h['page_end']}")
