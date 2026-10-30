from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import CHUNK_SIZE


class Question(BaseModel):
    question: str = Field(max_length=2000)
    k: int = Field(default=4, ge=1, le=8)
    source_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be empty")
        return value.strip()


class Chunk(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_index: int = Field(ge=0)
    token_count: int = Field(gt=0, le=CHUNK_SIZE)


class SearchMatch(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_index: int = Field(ge=0)
    score: float = Field(allow_inf_nan=False)

    @field_validator("id", "text", "source")
    @classmethod
    def field_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Document fields must not be blank")
        return value


ErrorCode = Literal[
    "legacy_unverified", "write_interrupted", "embedding_failed", "upsert_failed",
    "cleanup_failed", "visibility_check_failed", "stored_data_mismatch",
]


class DocumentSummary(BaseModel):
    source_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: str = Field(min_length=1)
    status: Literal["indexing", "ready", "failed"]
    chunk_count: int = Field(ge=0)
    error_code: ErrorCode | None = None


class UploadResult(DocumentSummary):
    duplicate: bool = False


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)
    insufficient_evidence: bool
    citations: list[str]

    @field_validator("answer")
    @classmethod
    def answer_not_blank(cls, value):
        if not value.strip():
            raise ValueError("Answer must not be empty")
        return value

    @model_validator(mode="after")
    def validate_citations(self):
        if any(not citation.strip() for citation in self.citations):
            raise ValueError("Citation IDs must not be empty")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("Citation IDs must be unique")
        if self.insufficient_evidence and self.citations:
            raise ValueError("Insufficient evidence must not include citations")
        if not self.insufficient_evidence and not self.citations:
            raise ValueError("Answer must include citations")
        return self


class SearchResult(BaseModel):
    matches: list[SearchMatch]


class AskResult(SearchResult):
    answer: Answer
