import json
import os
from hashlib import sha256
from urllib.parse import urlsplit


MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 320
CHUNK_OVERLAP = 48
EMBEDDING_DIMENSION = 384
EMBEDDING_FIELD = "embedding"
MAX_FILE_BYTES = 1024 * 1024
MAX_DOCUMENTS = 10
NAMESPACE = "doc_query"
PROCESSING = {
    "model": MODEL_NAME,
    "dimension": EMBEDDING_DIMENSION,
    "embedding_field": EMBEDDING_FIELD,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "cleaning": "preserve-fences-v2",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "normalize_embeddings": True,
}
CONFIG_ID = sha256(json.dumps(PROCESSING, sort_keys=True).encode()).hexdigest()
LEGACY_CONFIG_ID = "multilingual-e5-small-320-48-v1"


def index_target():
    host = os.environ.get("PINECONE_INDEX_HOST", "").strip().rstrip("/")
    namespace = os.environ.get("PINECONE_NAMESPACE", "")
    parsed = urlsplit(host if "://" in host else "https://" + host)
    if (not parsed.hostname or parsed.scheme != "https" or parsed.username
            or parsed.password or parsed.path or parsed.query or parsed.fragment
            or namespace != NAMESPACE):
        raise ValueError("Pinecone configuration is missing or invalid")
    return parsed.netloc.lower(), namespace
