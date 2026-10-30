import math
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version

from pinecone import DenseVectorQuery, Pinecone, RetryConfig
from pinecone.errors import ApiError, PineconeConnectionError, PineconeTimeoutError, ResponseParsingError
from pydantic import ValidationError

from app.config import CONFIG_ID, EMBEDDING_DIMENSION, EMBEDDING_FIELD, index_target
from app.errors import ServiceError
from app.schemas import SearchMatch


BATCH_SIZE = 50
REQUEST_TIMEOUT = 5.0
VISIBILITY_ATTEMPTS = 4
HTTP_TIMEOUT_PHASES = 4
logger = logging.getLogger(__name__)


class StoredDataMismatch(Exception):
    pass


class ChunkWriteError(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(error.detail)


def validate_configuration():
    try:
        target = index_target()
    except ValueError:
        raise ServiceError("pinecone_configuration", "invalid_configuration", 503,
                           "Pinecone target configuration is missing or invalid") from None
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ServiceError("pinecone_configuration", "missing_configuration", 503,
                           "Pinecone API key is missing")
    return target


def provider_error(exc, operation):
    status = exc.status_code if isinstance(exc, ApiError) else None
    if isinstance(exc, TimeoutError) or status in {408, 504}:
        return ServiceError(operation, "timeout", 504, "Pinecone request timed out")
    if isinstance(exc, ResponseParsingError):
        return ServiceError(operation, "invalid_response", 502, "Pinecone returned invalid data")
    code = f"http_{status}" if status is not None else "connection_failed"
    return ServiceError(operation, code, 503, "Pinecone is unavailable or its configuration was rejected")


@contextmanager
def connect_index(operation="pinecone_documents", retries=0):
    if retries not in {0, 1}:
        raise ValueError("Pinecone retries must be zero or one")
    host, _ = validate_configuration()
    if version("pinecone") != "10.2.0":
        raise ServiceError(operation, "unsupported_sdk", 503,
                           "The installed Pinecone SDK retry configuration must be verified")
    try:
        with Pinecone(api_key=os.environ["PINECONE_API_KEY"].strip(), timeout=REQUEST_TIMEOUT,
                      retry_config=RetryConfig(max_retries=retries, max_wait=1.0)) as client:
            with client.Index(host=host) as index:
                http_client = getattr(getattr(index, "_http", None), "_client", None)
                transport = getattr(http_client, "_transport", None)
                retry_config = getattr(transport, "_config", None)
                if (type(transport).__module__ != "pinecone._internal.http_client"
                        or type(transport).__name__ != "_RetryTransport"
                        or not isinstance(retry_config, RetryConfig)):
                    raise ServiceError(operation, "unsupported_transport", 503,
                                       "The Pinecone transport retry configuration must be verified")
                transport._config = replace(retry_config, max_retries=retries, max_wait=1.0)
                yield index
    except (ApiError, PineconeConnectionError, PineconeTimeoutError, ResponseParsingError) as exc:
        raise provider_error(exc, operation) from None


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
                raise ChunkWriteError(ServiceError("pinecone_upsert", "incomplete_write", 502,
                                                   "Pinecone did not accept the complete batch"))
        except (ApiError, PineconeConnectionError, PineconeTimeoutError, ResponseParsingError) as exc:
            raise ChunkWriteError(provider_error(exc, "pinecone_upsert")) from None


def cleanup_chunks(index, ids, namespace):
    for start in range(0, len(ids), BATCH_SIZE):
        try:
            index.documents.delete(namespace=namespace, ids=ids[start:start + BATCH_SIZE], timeout=REQUEST_TIMEOUT)
        except (ApiError, PineconeConnectionError, PineconeTimeoutError, ResponseParsingError) as exc:
            raise provider_error(exc, "pinecone_cleanup") from None


def wait_until_visible(index, chunks, namespace, config_id=CONFIG_ID, timeout=20):
    deadline = time.monotonic() + timeout
    pending = {chunk["id"]: chunk for chunk in chunks}
    if not pending:
        return False
    for attempt in range(VISIBILITY_ATTEMPTS):
        ids = list(pending)
        for start in range(0, len(ids), BATCH_SIZE):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            batch = ids[start:start + BATCH_SIZE]
            response = index.documents.fetch(
                namespace=namespace, ids=batch,
                include_fields=["text", "source", "source_id", "chunk_index", "token_count", "config"],
                timeout=min(REQUEST_TIMEOUT, remaining / HTTP_TIMEOUT_PHASES),
            )
            for document in response.documents.values():
                document_id = document.get("_id")
                if document_id not in batch or document_id not in pending:
                    raise StoredDataMismatch()
                chunk = pending[document_id]
                if (document.get("config") != config_id
                        or any(document.get(key) != chunk[key] for key in
                               ("text", "source", "source_id", "chunk_index", "token_count"))):
                    raise StoredDataMismatch()
                del pending[document_id]
        if not pending:
            return time.monotonic() <= deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if attempt + 1 < VISIBILITY_ATTEMPTS:
            time.sleep(min(1, remaining))
    return False


def search(index, vector: list[float], k: int, ready_sources: dict):
    if not 1 <= k <= 8:
        raise ValueError("k must be between 1 and 8")
    _, namespace = index_target()
    if not ready_sources:
        return []
    response = index.documents.search(
        namespace=namespace, top_k=k,
        filter={"source_id": {"$in": list(ready_sources)}},
        score_by=[DenseVectorQuery(field=EMBEDDING_FIELD, values=vector)],
        include_fields=["text", "source", "source_id", "chunk_index", "config"],
        timeout=REQUEST_TIMEOUT,
    )
    matches = []
    seen = set()
    for doc in response.matches:
        fields = {key: doc.get(key) for key in ("text", "source", "source_id", "chunk_index")}
        chunk_index = fields["chunk_index"]
        if isinstance(chunk_index, float) and math.isfinite(chunk_index) and chunk_index.is_integer():
            fields["chunk_index"] = int(chunk_index)
        try:
            match = SearchMatch.model_validate({"id": doc.get("_id"), "score": doc.get("_score"), **fields}, strict=True)
        except ValidationError as exc:
            failures = ",".join(f"{error['loc'][0]}:{error['type']}" for error in exc.errors())
            logger.error("operation=pinecone_search code=invalid_record fields=%s status=502", failures)
            raise ServiceError("pinecone_search", "invalid_record", 502,
                               "Pinecone returned an incomplete or invalid document record") from None
        entry = ready_sources.get(match.source_id)
        if (entry is None or doc.get("config") != entry["config_id"]
                or match.chunk_index >= len(entry["chunks"]) or match.id in seen):
            raise ServiceError("pinecone_search", "incompatible_record", 502,
                               "Pinecone returned a document incompatible with the ready manifest")
        expected = entry["chunks"][match.chunk_index]
        if any(getattr(match, key) != expected[key] for key in ("id", "text", "source", "source_id", "chunk_index")):
            raise ServiceError("pinecone_search", "mismatched_record", 502,
                               "Pinecone returned a document that differs from the ready manifest")
        seen.add(match.id)
        matches.append(match.model_dump())
    return matches
