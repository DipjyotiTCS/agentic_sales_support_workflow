import os
from typing import Dict, List
from tqdm import tqdm
from dotenv import load_dotenv

from chroma_client import get_chroma_client, get_collection
from embeddings import embed_texts, DEFAULT_EMBED_MODEL
from pdf_parser import extract_pdf_pages, parse_qa_from_pages
from utils import sha1_of_file, chunk_text, safe_list_pdfs

def build_embedding_input(section: str, question: str, answer_chunk: str) -> str:
    parts = []
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Q: {question}")
    parts.append(f"A: {answer_chunk}")
    return "\n".join(parts).strip()

def ingest_pdf(pdf_path: str) -> Dict:
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "support-kb")
    embed_model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    client = get_chroma_client(persist_dir=persist_dir)
    collection = get_collection(client, name=collection_name, metadata={"hnsw:space": "cosine"})

    source_name = os.path.basename(pdf_path)
    doc_id = sha1_of_file(pdf_path)

    pages = extract_pdf_pages(pdf_path)
    qa_records = parse_qa_from_pages(pages, source_name=source_name)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for i, r in enumerate(qa_records):
        chunks = chunk_text(r["answer"], max_chars=2200, overlap=250)
        for j, ch in enumerate(chunks):
            cid = f"{doc_id}-qa{i}-c{j}"
            doc = build_embedding_input(r.get("section") or "", r["question"], ch)
            meta = {
                "doc_id": doc_id,
                "source": r["source"],
                "section": r.get("section") or "",
                "question": r["question"],
                "answer_chunk": ch,
                "chunk_index": j,
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
                "reference_urls": " | ".join(r.get("reference_urls", [])),
                "embedding_model": embed_model,
            }
            ids.append(cid)
            documents.append(doc)
            metadatas.append(meta)

    if not documents:
        return {
            "collection": collection_name,
            "persist_dir": persist_dir,
            "source": source_name,
            "doc_id": doc_id,
            "qa_pairs": 0,
            "vectors": 0,
            "warning": "No Q/A pairs found. Check PDF markers (Q:/A:) and text extraction.",
        }

    batch_size = 64
    embeddings: List[List[float]] = []
    for start in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
        batch_docs = documents[start:start+batch_size]
        embeddings.extend(embed_texts(batch_docs, model=embed_model))

    # Idempotent replace for this doc_id
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return {
        "collection": collection_name,
        "persist_dir": persist_dir,
        "source": source_name,
        "doc_id": doc_id,
        "qa_pairs": len(qa_records),
        "vectors": len(ids),
        "embedding_model": embed_model,
    }


def ingest_pdf_folder(pdf_dir: str) -> Dict:
    """Ingest all PDFs from a folder into the configured Chroma collection."""
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "support-kb")
    embed_model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    client = get_chroma_client(persist_dir=persist_dir)
    collection = get_collection(client, name=collection_name, metadata={"hnsw:space": "cosine"})

    pdfs = safe_list_pdfs(pdf_dir)
    docs_out = []
    total_vectors = 0
    total_qas = 0

    for pdf_path in pdfs:
        # Reuse ingest_pdf logic but avoid creating client each time by inlining minimal bits
        source_name = os.path.basename(pdf_path)
        doc_id = sha1_of_file(pdf_path)

        pages = extract_pdf_pages(pdf_path)
        qa_records = parse_qa_from_pages(pages, source_name=source_name)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for i, r in enumerate(qa_records):
            chunks = chunk_text(r["answer"], max_chars=2200, overlap=250)
            for j, ch in enumerate(chunks):
                cid = f"{doc_id}-qa{i}-c{j}"
                doc = build_embedding_input(r.get("section") or "", r["question"], ch)
                meta = {
                    "doc_id": doc_id,
                    "source": r["source"],
                    "section": r.get("section") or "",
                    "question": r["question"],
                    "answer_chunk": ch,
                    "chunk_index": j,
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                    "reference_urls": " | ".join(r.get("reference_urls", [])),
                    "embedding_model": embed_model,
                }
                ids.append(cid)
                documents.append(doc)
                metadatas.append(meta)

        if not documents:
            docs_out.append({"source": source_name, "doc_id": doc_id, "qa_pairs": 0, "vectors": 0, "warning": "No Q/A pairs found"})
            continue

        # Embed in batches
        batch_size = 64
        embeddings: List[List[float]] = []
        for start in tqdm(range(0, len(documents), batch_size), desc=f"Embedding {source_name}"):
            batch_docs = documents[start:start+batch_size]
            embeddings.extend(embed_texts(batch_docs, model=embed_model))

        # Idempotent replace for this doc_id
        try:
            collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass

        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        docs_out.append({"source": source_name, "doc_id": doc_id, "qa_pairs": len(qa_records), "vectors": len(ids)})
        total_vectors += len(ids)
        total_qas += len(qa_records)

    return {
        "collection": collection_name,
        "persist_dir": persist_dir,
        "pdf_dir": pdf_dir,
        "pdf_count": len(pdfs),
        "total_qa_pairs": total_qas,
        "total_vectors": total_vectors,
        "documents": docs_out,
    }


if __name__ == "__main__":
    load_dotenv()
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")
        raise SystemExit(1)
    print(ingest_pdf(sys.argv[1]))
