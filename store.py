import os

from pinecone import DenseVectorQuery, Pinecone


def connect_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_INDEX_HOST")
    if not api_key or not host:
        raise RuntimeError("Pinecone configuration is missing")
    return Pinecone(api_key=api_key).Index(host=host)


def search(index, vector: list[float], k: int = 4):
    if not 1 <= k <= 8:
        raise ValueError("k must be between 1 and 8")
    namespace = os.environ.get("PINECONE_NAMESPACE", "doc_query")
    return index.documents.search(
        namespace=namespace,
        top_k=k,
        score_by=[DenseVectorQuery(field="embedding", values=vector)],
        include_fields=["text", "source"],
    )
