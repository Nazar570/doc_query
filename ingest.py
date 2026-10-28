import os


MAX_FILE_BYTES = 1024 * 1024


def prepare_document(filename: str, content: bytes) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if os.path.splitext(name)[1].lower() not in {".md", ".txt"}:
        raise ValueError("Dozwolone są wyłącznie pliki .md i .txt")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Plik przekracza limit 1 MiB")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Plik musi być zakodowany w UTF-8") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Plik jest pusty")
    return text
