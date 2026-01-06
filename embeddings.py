import os
from typing import List
from openai import OpenAI

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536

_client = None

def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client

def embed_texts(texts: List[str], model: str = DEFAULT_EMBED_MODEL) -> List[List[float]]:
    client = get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
