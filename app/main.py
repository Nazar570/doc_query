import logging
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import CONFIG_ID, MAX_DOCUMENTS, index_target
from app.schemas import AskResult, Chunk, DocumentSummary, Question, SearchResult, UploadResult
from app.services import ingest, manifest, rag, store
from app.services.embeddings import Embeddings


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI()
write_lock = Lock()
embedding_lock = Lock()
_embeddings = None
logger = logging.getLogger(__name__)


def get_embeddings():
    global _embeddings
    with embedding_lock:
        if _embeddings is None:
            _embeddings = Embeddings()
        return _embeddings


def service_error():
    logger.error("A required service or configuration is unavailable")
    return HTTPException(status_code=503, detail="A required service or configuration is unavailable")


@app.exception_handler(manifest.ManifestError)
def manifest_error(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def document_result(entry, response, duplicate=False):
    if entry["status"] == "indexing":
        response.status_code = 202
    elif entry["status"] == "failed":
        response.status_code = 503
    return UploadResult(**DocumentSummary.model_validate(entry).model_dump(), duplicate=duplicate)


def refresh_entry(data, entry):
    if entry["status"] == "ready":
        return
    if not entry["legacy"] and entry["status"] == "failed":
        raise HTTPException(status_code=409, detail="Reupload the same file to retry the failed import")
    if entry["error_code"] == "write_interrupted":
        entry.update(status="failed")
        manifest.save(data)
        return
    try:
        target = index_target()
        index = store.connect_index()
        visible = store.wait_until_visible(index, entry["chunks"], target[1], entry["config_id"])
    except store.StoredDataMismatch:
        entry.update(status="failed", error_code="stored_data_mismatch")
    except Exception:
        entry["error_code"] = "visibility_check_failed"
        logger.error("Document visibility check failed")
        manifest.save(data)
        raise service_error() from None
    else:
        if visible:
            data.update(host=target[0], namespace=target[1])
            entry.update(status="ready", error_code=None)
        else:
            entry.update(status="indexing", error_code="legacy_unverified" if entry["legacy"] else None)
    manifest.save(data)


@app.get("/")
def homepage():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResult)
def upload_document(response: Response, file: UploadFile = File(...)):
    content = file.file.read(ingest.MAX_FILE_BYTES + 1)
    try:
        text = ingest.prepare_document(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    source_id = ingest.document_id(content)
    source = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1]
    source = "".join(c for c in source if c.isprintable())[:255] or "document"
    with write_lock:
        data = manifest.load()
        documents = data["documents"]
        existing = documents.get(source_id)
        if existing and (existing["status"] != "failed" or existing["legacy"]):
            refresh_entry(data, existing)
            return document_result(existing, response, duplicate=True)
        active = sum(doc["status"] != "failed" for doc in documents.values())
        if active >= MAX_DOCUMENTS:
            raise HTTPException(status_code=400, detail="Document limit reached; failed imports do not reserve a slot")
        try:
            target = index_target()
            index = store.connect_index()
        except Exception as exc:
            raise service_error() from exc
        data.update(host=target[0], namespace=target[1])
        entry = existing or {"source_id": source_id, "source": source, "status": "failed",
                             "chunk_count": 0, "chunks": [], "error_code": "embedding_failed",
                             "config_id": CONFIG_ID, "legacy": False}
        documents[source_id] = entry
        if existing and entry["error_code"] in {"cleanup_failed", "write_interrupted"}:
            try:
                store.cleanup_chunks(index, [c["id"] for c in entry["chunks"]], target[1])
            except Exception:
                entry.update(status="failed", error_code="cleanup_failed")
                manifest.save(data)
                return document_result(entry, response)
        try:
            model = get_embeddings()
            if not entry["chunks"]:
                pieces = ingest.split_document(text, model.model.tokenizer)
                entry["chunks"] = [
                    {"id": f"{source_id}-{i}", "text": part, "source": entry["source"],
                     "source_id": source_id, "chunk_index": i,
                     "token_count": len(model.model.tokenizer.encode(part, add_special_tokens=False))}
                    for i, part in enumerate(pieces)
                ]
                entry["chunk_count"] = len(entry["chunks"])
            if not entry["chunks"]:
                raise ValueError("Document contains no chunks")
            vectors = model.passages([c["text"] for c in entry["chunks"]])
        except Exception:
            entry.update(status="failed", error_code="embedding_failed")
            manifest.save(data)
            return document_result(entry, response)
        entry.update(status="indexing", error_code="write_interrupted")
        manifest.save(data)
        try:
            store.write_chunks(index, entry["chunks"], vectors, target[1])
        except store.ChunkWriteError:
            entry.update(status="failed", error_code="upsert_failed")
            try:
                store.cleanup_chunks(index, [c["id"] for c in entry["chunks"]], target[1])
            except Exception:
                entry["error_code"] = "cleanup_failed"
            manifest.save(data)
            return document_result(entry, response)
        except (ValueError, TypeError):
            entry.update(status="failed", error_code="embedding_failed")
            manifest.save(data)
            return document_result(entry, response)
        entry["error_code"] = None
        manifest.save(data)
        refresh_entry(data, entry)
        return document_result(entry, response)


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents():
    return list(manifest.load()["documents"].values())


@app.get("/documents/{source_id}/chunks", response_model=list[Chunk])
def list_chunks(source_id: str):
    document = manifest.load()["documents"].get(source_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document["chunks"]


@app.post("/documents/{source_id}/refresh", response_model=UploadResult)
def refresh_document(source_id: str, response: Response):
    with write_lock:
        data = manifest.load()
        document = data["documents"].get(source_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        refresh_entry(data, document)
        return document_result(document, response, duplicate=True)


def search_question(request: Question):
    data = manifest.load()
    try:
        index = store.connect_index()
        vector = get_embeddings().query(request.question)
        found = store.search(index, vector, request.k)
        ready = {key for key, doc in data["documents"].items() if doc["status"] == "ready"}
        return [chunk for chunk in found if chunk["source_id"] in ready]
    except Exception as exc:
        raise service_error() from exc


@app.post("/search", response_model=SearchResult)
def search(request: Question):
    return {"matches": search_question(request)}


@app.post("/ask", response_model=AskResult)
def ask(request: Question):
    matches = search_question(request)
    try:
        answer = rag.generate_answer(request.question, matches)
    except Exception as exc:
        raise service_error() from exc
    return {"answer": answer.model_dump(), "matches": matches}
