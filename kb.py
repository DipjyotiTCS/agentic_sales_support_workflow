import os, json, time
from typing import List, Tuple, Optional
import numpy as np
from pypdf import PdfReader
from db import conn_ctx
from llm import get_embeddings

def ensure_kb_tables():
    with conn_ctx() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS kb_documents(
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            uploaded_at TEXT
        );
        CREATE TABLE IF NOT EXISTS kb_chunks(
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_text TEXT,
            embedding_json TEXT,
            FOREIGN KEY(doc_id) REFERENCES kb_documents(doc_id)
        );
        """)

def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c.strip()]

def ingest_pdf(file_path: str, filename: str) -> Tuple[int, int]:
    """
    Returns (doc_id, num_chunks).
    Stores embeddings if OpenAI key exists; else stores NULL embeddings.
    """
    ensure_kb_tables()
    reader = PdfReader(file_path)
    pages_text = []
    for p in reader.pages:
        pages_text.append(p.extract_text() or "")
    full_text = "\n".join(pages_text).strip()
    chunks = _chunk_text(full_text)

    emb = get_embeddings()
    vectors = None
    if emb is not None and chunks:
        vectors = emb.embed_documents(chunks)

    with conn_ctx() as conn:
        cur = conn.execute(
            "INSERT INTO kb_documents(filename, uploaded_at) VALUES(?, ?)",
            (filename, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        doc_id = cur.lastrowid
        for idx, ch in enumerate(chunks):
            ejson = None
            if vectors is not None:
                ejson = json.dumps(vectors[idx])
            conn.execute(
                "INSERT INTO kb_chunks(doc_id, chunk_text, embedding_json) VALUES(?,?,?)",
                (doc_id, ch, ejson),
            )
    return doc_id, len(chunks)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def retrieve_context(query: str, k: int = 4) -> List[str]:
    """
    If embeddings exist, does cosine similarity retrieval.
    If not, falls back to naive keyword overlap scoring.
    """
    ensure_kb_tables()
    with conn_ctx() as conn:
        rows = conn.execute("SELECT chunk_text, embedding_json FROM kb_chunks").fetchall()

    if not rows:
        return []

    emb = get_embeddings()
    # Path A: embedding-based retrieval
    if emb is not None and any(r["embedding_json"] for r in rows):
        qv = np.array(emb.embed_query(query), dtype=np.float32)
        scored = []
        for r in rows:
            if not r["embedding_json"]:
                continue
            v = np.array(json.loads(r["embedding_json"]), dtype=np.float32)
            scored.append((_cosine(qv, v), r["chunk_text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    # Path B: keyword overlap fallback (demo mode)
    q = set(query.lower().split())
    scored = []
    for r in rows:
        w = set((r["chunk_text"] or "").lower().split())
        score = len(q.intersection(w)) / max(1, len(q))
        scored.append((score, r["chunk_text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k] if t.strip()]


def search_kb(query: str, top_k: int = 4) -> List[dict]:
    """Search KB chunks relevant to query. Returns list of dicts with filename, chunk_id, score, text."""
    ensure_kb_tables()
    query = (query or "").strip()
    if not query:
        return []
    emb_model = get_embeddings()
    q_emb = None
    if emb_model is not None:
        try:
            q_emb = emb_model.embed_query(query)
        except Exception:
            q_emb = None

    with conn_ctx() as conn:
        rows = conn.execute(
            """SELECT c.chunk_id, c.doc_id, c.chunk_text, c.embedding_json, d.filename
               FROM kb_chunks c JOIN kb_documents d ON c.doc_id = d.doc_id"""
        ).fetchall()

    results = []
    if q_emb is None:
        # Keyword fallback scoring
        q_words = set(query.lower().split())
        for r in rows:
            text = (r[2] or "").lower()
            overlap = sum(1 for w in q_words if w in text)
            score = overlap / max(1, len(q_words))
            if score > 0:
                results.append({
                    "chunk_id": r[0],
                    "doc_id": r[1],
                    "filename": r[4],
                    "score": float(score),
                    "text": r[2],
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # Embedding cosine similarity
    q = np.array(q_emb, dtype=float)
    qn = float(np.linalg.norm(q) + 1e-9)
    for r in rows:
        try:
            v = np.array(json.loads(r[3] or "[]"), dtype=float)
            if v.size == 0:
                continue
            score = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-9)))
            results.append({
                "chunk_id": r[0],
                "doc_id": r[1],
                "filename": r[4],
                "score": score,
                "text": r[2],
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
