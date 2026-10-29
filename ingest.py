import os
from hashlib import sha256

from langchain_text_splitters import RecursiveCharacterTextSplitter



MAX_FILE_BYTES = 1024 * 1024


def document_id(content: bytes) -> str:
    return sha256(content).hexdigest()


def split_document(text: str, tokenizer) -> list[str]:
    def token_count(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=320,
        chunk_overlap=48,
        length_function=token_count,
    )
    chunks = splitter.split_text(text)
    if any(token_count(chunk) > 320 for chunk in chunks):
        raise ValueError("Chunk exceeds the limit of 320 tokens")
    return chunks


def prepare_document(filename: str, content: bytes) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if os.path.splitext(name)[1].lower() not in {".md", ".txt"}:
        raise ValueError("Only .md and .txt files are allowed")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the limit of 1 MiB")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("File must be encoded in UTF-8") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("File is empty")
    return text
