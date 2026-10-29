import os
import time

from pinecone import DenseVectorQuery, Pinecone


def connect_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_INDEX_HOST")
    namespace = os.environ.get("PINECONE_NAMESPACE")
    if not api_key or not host or namespace != "doc_query":
        raise RuntimeError("Pinecone configuration is missing or namespace is invalid")
    return Pinecone(api_key=api_key).Index(host=host)


def write_chunks(index, chunks, vectors):
    documents = [
        {"_id": chunk["id"], "embedding": vector, "text": chunk["text"],
         "source": chunk["source"], "source_id": chunk["source_id"],
         "chunk_index": chunk["chunk_index"], "token_count": chunk["token_count"],
         "config": "multilingual-e5-small-320-48-v1"}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(documents), 50):
        index.documents.upsert(namespace="doc_query", documents=documents[start:start + 50])


def wait_until_visible(index, ids, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = 0
        for start in range(0, len(ids), 50):
            response = index.documents.fetch(namespace="doc_query", ids=ids[start:start + 50])
            found += len(response.documents)
        if found == len(ids):
            return True
        time.sleep(1)
    return False


def search(index, vector: list[float], k: int = 4):
    if not 1 <= k <= 8:
        raise ValueError("k must be between 1 and 8")
    response = index.documents.search(
        namespace="doc_query", top_k=k,
        score_by=[DenseVectorQuery(field="embedding", values=vector)],
        include_fields=["text", "source", "source_id", "chunk_index"],
    )
    return [{"id": doc.id, "score": doc.score, "text": doc.text, "source": doc.source,
             "source_id": doc.source_id, "chunk_index": doc.chunk_index}
            for doc in response.matches]
