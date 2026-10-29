import json
import os
from pathlib import Path


PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.json"


def load():
    if not PATH.exists():
        return {}
    return json.loads(PATH.read_text(encoding="utf-8"))


def save(documents):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, PATH)
