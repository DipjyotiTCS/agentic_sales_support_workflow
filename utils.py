import os
import hashlib
import re
from typing import List

PDF_URL_GLUE_CHAR = "￾"
URL_RE = re.compile(r"(https?://\S+)")

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_text(s: str) -> str:
    s = s.replace("\r", "")
    s = s.replace(PDF_URL_GLUE_CHAR, "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(https?://[^\s\n]+)\n([^\s]+)", r"\1\2", s)
    return s.strip()

def extract_urls(s: str) -> List[str]:
    s = (s or "").replace(PDF_URL_GLUE_CHAR, "")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(https?://[^\s\n]+)\n([^\s]+)", r"\1\2", s)
    urls = URL_RE.findall(s)
    seen = set()
    out = []
    for u in urls:
        u = u.rstrip(").,;")
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def chunk_text(text: str, max_chars: int = 2200, overlap: int = 250) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def safe_list_pdfs(pdf_dir: str):
    """Return sorted list of PDF file paths in a directory; [] if folder missing."""
    try:
        return [os.path.join(pdf_dir, n) for n in sorted(os.listdir(pdf_dir)) if n.lower().endswith(".pdf")]
    except FileNotFoundError:
        return []
