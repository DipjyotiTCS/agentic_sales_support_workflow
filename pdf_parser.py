from typing import Dict, List, Optional
import fitz  # PyMuPDF
from utils import normalize_text, extract_urls

BULLET_PREFIXES = ("•", "-", "*")

def is_section_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    lower = s.lower()
    if lower.startswith("q:") or lower.startswith("a:") or lower.startswith("more details"):
        return False
    if s.startswith(BULLET_PREFIXES):
        return False
    if "http://" in s or "https://" in s:
        return False
    return len(s) <= 100

def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        pages.append({"page": i + 1, "text": normalize_text(txt)})
    return pages

def parse_qa_from_pages(pages: List[Dict], source_name: str) -> List[Dict]:
    records: List[Dict] = []
    current_section: Optional[str] = None

    state = "SEEK_Q"
    q_lines: List[str] = []
    a_lines: List[str] = []
    page_start = None
    page_end = None

    def flush_record():
        nonlocal q_lines, a_lines, page_start, page_end, state
        q = " ".join([l.strip() for l in q_lines]).strip()
        a = "\n".join([l.rstrip() for l in a_lines]).strip()
        if q and a:
            urls = extract_urls(a)
            records.append({
                "section": current_section,
                "question": q,
                "answer": a,
                "reference_urls": urls,
                "source": source_name,
                "page_start": page_start,
                "page_end": page_end,
            })
        q_lines = []
        a_lines = []
        page_start = None
        page_end = None
        state = "SEEK_Q"

    for p in pages:
        for raw in p["text"].splitlines():
            line = raw.strip()

            if state == "SEEK_Q" and is_section_header(line):
                current_section = line
                continue

            if line.lower().startswith("q:"):
                if state in ("IN_Q", "IN_A"):
                    flush_record()
                state = "IN_Q"
                page_start = page_start or p["page"]
                page_end = p["page"]
                q_lines.append(line[2:].strip())
                continue

            if line.lower().startswith("a:"):
                state = "IN_A"
                page_start = page_start or p["page"]
                page_end = p["page"]
                a_lines.append(line[2:].strip())
                continue

            if state == "IN_Q":
                if line:
                    q_lines.append(line)
                page_end = p["page"]
                continue

            if state == "IN_A":
                a_lines.append(raw.rstrip())
                page_end = p["page"]
                continue

    if state in ("IN_Q", "IN_A"):
        flush_record()

    return records
