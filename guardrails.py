import re
from typing import Tuple, List

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"you are now",
    r"act as",
    r"jailbreak",
    r"do anything now",
]

def sanitize_text(s: str) -> str:
    # Strip common dangerous control chars; keep it readable
    s = s.replace("\x00", "")
    return s.strip()

def basic_injection_check(text: str) -> Tuple[bool, List[str]]:
    hits = []
    low = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            hits.append(pat)
    return (len(hits) > 0, hits)

def guard_email_input(subject: str, body: str) -> Tuple[bool, List[str]]:
    flags = []
    bad, hits = basic_injection_check(subject + "\n" + body)
    if bad:
        flags.append(f"Possible prompt-injection patterns detected: {hits}")
    if len(body) > 20000:
        flags.append("Body too large; truncated.")
    return (len(flags) == 0, flags)
