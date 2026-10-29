import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import CONFIG_ID, LEGACY_CONFIG_ID, PROCESSING, index_target
from app.schemas import Chunk, DocumentSummary


PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.json"


class ManifestError(Exception):
    pass


class DocumentEntry(DocumentSummary):
    model_config = ConfigDict(extra="forbid")
    chunks: list[Chunk]
    config_id: str
    legacy: bool = False


class ManifestData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[2]
    config_id: str
    processing: dict
    host: str | None
    namespace: str | None
    documents: dict[str, DocumentEntry]


def empty(target=None):
    return {"version": 2, "config_id": CONFIG_ID, "processing": dict(PROCESSING),
            "host": target[0] if target else None,
            "namespace": target[1] if target else None, "documents": {}}


def validate(data):
    try:
        parsed = ManifestData.model_validate(data, strict=True)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ManifestError("Manifest is invalid; restore its backup or review it before continuing") from exc
    if parsed.config_id != CONFIG_ID or parsed.processing != PROCESSING:
        raise ManifestError("Manifest processing configuration differs; an explicit reindexing decision is required")
    if (parsed.host is None) != (parsed.namespace is None):
        raise ManifestError("Manifest index binding is invalid")
    for source_id, entry in parsed.documents.items():
        if entry.source_id != source_id or entry.chunk_count != len(entry.chunks):
            raise ManifestError("Manifest document identity or chunk count is invalid")
        expected_config = LEGACY_CONFIG_ID if entry.legacy else CONFIG_ID
        if entry.config_id != expected_config:
            raise ManifestError("Manifest document configuration is incompatible")
        if entry.status == "ready" and (not entry.chunks or entry.error_code or parsed.host is None):
            raise ManifestError("Manifest ready state is invalid")
        if entry.status == "indexing" and not entry.chunks:
            raise ManifestError("Manifest indexing state has no chunks")
        if entry.status == "failed" and not entry.error_code:
            raise ManifestError("Manifest failed state has no error code")
        for i, chunk in enumerate(entry.chunks):
            if (chunk.id != f"{source_id}-{i}" or chunk.source_id != source_id
                    or chunk.chunk_index != i or chunk.source != entry.source):
                raise ManifestError("Manifest chunk identity is invalid")
    return parsed.model_dump()


def load():
    if not PATH.exists():
        return empty()
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError("Manifest cannot be read; restore its backup or review it before continuing") from exc
    if isinstance(data, dict) and "version" not in data:
        raise ManifestError("Legacy manifest requires local migration: python -m app.services.manifest")
    data = validate(data)
    if data["host"] is not None:
        try:
            target = index_target()
        except ValueError as exc:
            raise ManifestError("Pinecone target configuration is required to verify the manifest binding") from exc
        if (data["host"], data["namespace"]) != target:
            raise ManifestError("Manifest index or namespace differs; an explicit reindexing decision is required")
    return data


def save(data):
    data = validate(data)
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, PATH)
    except OSError as exc:
        raise ManifestError("Manifest could not be saved; check local file access before retrying") from exc


def migrate_legacy():
    try:
        original = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManifestError("Legacy manifest cannot be read") from exc
    if not isinstance(original, dict) or "version" in original:
        raise ManifestError("Expected an unversioned legacy manifest; no changes were made")
    data = empty()
    for source_id, document in original.items():
        if not isinstance(document, dict):
            raise ManifestError("Legacy manifest document is invalid")
        try:
            DocumentSummary.model_validate(document, strict=True)
        except ValidationError as exc:
            raise ManifestError("Legacy manifest document summary is invalid") from exc
        entry = dict(document)
        entry.update(status="indexing", error_code="legacy_unverified",
                     config_id=LEGACY_CONFIG_ID, legacy=True)
        data["documents"][source_id] = entry
    data = validate(data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = PATH.with_name(f"manifest.backup-{stamp}.json")
    try:
        shutil.copy2(PATH, backup)
    except OSError as exc:
        raise ManifestError("Manifest backup failed; migration was cancelled") from exc
    save(data)
    return backup


if __name__ == "__main__":
    try:
        backup = migrate_legacy()
    except ManifestError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Manifest migrated. Backup: {backup}. Refresh verifies legacy records before they become ready.")
