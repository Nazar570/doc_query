import os
import re
from hashlib import sha256

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_BYTES


def document_id(content: bytes) -> str:
    return sha256(content).hexdigest()


def split_document(text: str, tokenizer) -> list[str]:
    def token_count(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_count,
    )
    chunks = splitter.split_text(text)
    if any(token_count(chunk) > CHUNK_SIZE for chunk in chunks):
        raise ValueError(f"Chunk exceeds the limit of {CHUNK_SIZE} tokens")
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
    lines = []
    fence = None
    blank_lines = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)", line)
        if marker:
            token = marker.group(1)
            tail = marker.group(2)
            if fence is None and (token[0] != "`" or "`" not in tail):
                fence = token
            elif (fence is not None and token[0] == fence[0]
                  and len(token) >= len(fence) and not tail.strip(" \t")):
                fence = None
            blank_lines = 0
            lines.append(line)
        elif fence is None and not line.strip():
            blank_lines += 1
            if blank_lines <= 2:
                lines.append(line)
        else:
            blank_lines = 0
            lines.append(line)
    return "".join(lines)
