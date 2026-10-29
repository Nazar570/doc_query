import math
import os
import time

from pinecone import DenseVectorQuery, Pinecone, RetryConfig

from app.config import CONFIG_ID, EMBEDDING_DIMENSION, EMBEDDING_FIELD, index_target


BATCH_SIZE = 50
REQUEST_TIMEOUT = 5.0
VISIBILITY_ATTEMPTS = 4
HTTP_TIMEOUT_PHASES = 4


class StoredDataMismatch(Exception):
    pass


class ChunkWriteError(Exception):
    pass


def connect_index():
    host, _ = index_target()
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Pinecone configuration is missing")
    return Pinecone(api_key=api_key, timeout=REQUEST_TIMEOUT,
                    retry_config=RetryConfig(max_retries=0)).Index(host=host)


def write_chunks(index, chunks, vectors, namespace):
    documents = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != EMBEDDING_DIMENSION or not all(math.isfinite(v) for v in vector):
            raise ValueError("Embedding vector has an invalid dimension or value")
        documents.append({"_id": chunk["id"], EMBEDDING_FIELD: vector,
                          "text": chunk["text"], "source": chunk["source"],
                          "source_id": chunk["source_id"], "chunk_index": chunk["chunk_index"],
                          "token_count": chunk["token_count"], "config": CONFIG_ID})
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        try:
            response = index.documents.upsert(namespace=namespace, documents=batch, timeout=REQUEST_TIMEOUT)
            if response.upserted_count != len(batch):
                raise RuntimeError("Pinecone did not accept the complete batch")
        except Exception as exc:
            raise ChunkWriteError() from exc


def cleanup_chunks(index, ids, namespace):
    for start in range(0, len(ids), BATCH_SIZE):
        index.documents.delete(namespace=namespace, ids=ids[start:start + BATCH_SIZE], timeout=REQUEST_TIMEOUT)


def wait_until_visible(index, chunks, namespace, config_id=CONFIG_ID, timeout=20):
    deadline = time.monotonic() + timeout
    expected = {chunk["id"]: chunk for chunk in chunks}
    if not expected:
        return False
    for attempt in range(VISIBILITY_ATTEMPTS):
        found = set()
        ids = list(expected)
        for start in range(0, len(ids), BATCH_SIZE):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            batch = ids[start:start + BATCH_SIZE]
            try:
                response = index.documents.fetch(
                    namespace=namespace, ids=batch,
                    include_fields=["text", "source", "source_id", "chunk_index", "token_count", "config"],
                    timeout=min(REQUEST_TIMEOUT, remaining / HTTP_TIMEOUT_PHASES),
                )
            except TimeoutError:
                return False
            for document in response.documents.values():
                if document.id not in batch:
                    raise StoredDataMismatch()
                chunk = expected[document.id]
                if (document.get("config") != config_id
                        or any(document.get(key) != chunk[key] for key in
                               ("text", "source", "source_id", "chunk_index", "token_count"))):
                    raise StoredDataMismatch()
                found.add(document.id)
        if found == set(expected):
            return time.monotonic() <= deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if attempt + 1 < VISIBILITY_ATTEMPTS:
            time.sleep(min(1, remaining))
    return False


def search(index, vector: list[float], k: int = 4):
    if not 1 <= k <= 8:
        raise ValueError("k must be between 1 and 8")
    _, namespace = index_target()
    response = index.documents.search(
        namespace=namespace, top_k=k,
        score_by=[DenseVectorQuery(field=EMBEDDING_FIELD, values=vector)],
        include_fields=["text", "source", "source_id", "chunk_index"],
    )
    return [{"id": doc.id, "score": doc.score, "text": doc.text, "source": doc.source,
             "source_id": doc.source_id, "chunk_index": doc.chunk_index}
            for doc in response.matches]
