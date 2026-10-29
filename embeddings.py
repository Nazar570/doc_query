from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

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
        return self._encode(texts, "passage: ")

    def query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Question must not be empty")
        return self._encode([text], "query: ")[0]
