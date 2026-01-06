import chromadb
from chromadb.config import Settings

def get_chroma_client(persist_dir: str):
    return chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

#def get_collection(client, name: str, metadata=None):
#    return client.get_or_create_collection(name=name, metadata=metadata or {})

def get_collection(client, name: str, metadata=None):
    if metadata is None:
        # Don't pass metadata at all
        return client.get_or_create_collection(name=name)
    # If caller passed metadata, ensure it's not empty
    if isinstance(metadata, dict) and len(metadata) == 0:
        metadata = {"hnsw:space": "cosine"}
    return client.get_or_create_collection(name=name, metadata=metadata)
