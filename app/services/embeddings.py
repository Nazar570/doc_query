from sentence_transformers import SentenceTransformer

from app.config import MODEL_NAME, PROCESSING


class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        inputs = [prefix + text for text in texts]
        for value in inputs:
            tokens = self.model.tokenizer.encode(value, add_special_tokens=True, truncation=False)
            if len(tokens) > self.model.max_seq_length:
                raise ValueError("Embedding input exceeds the model token limit")
        if not inputs:
            return []
        return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, PROCESSING["passage_prefix"])

    def query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Question must not be empty")
        return self._encode([text], PROCESSING["query_prefix"])[0]
