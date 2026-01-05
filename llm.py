import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def get_chat_model() -> Optional[ChatOpenAI]:
    if not has_openai_key():
        return None
    # Use a lightweight, fast model for demo. Change as needed.
    return ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)

def get_embeddings() -> Optional[OpenAIEmbeddings]:
    if not has_openai_key():
        return None
    return OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
