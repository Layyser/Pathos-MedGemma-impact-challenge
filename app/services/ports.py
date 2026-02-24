from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence

from app.domain.chunk import Chunk
from app.domain.dataset import Dataset
from app.domain.document import Document
from app.domain.embedding import Embedding
from app.domain.knowledge import KnowledgeSnippet


class ParserPort(Protocol):
    def parse_text(self, file_path: Path) -> str:
        ...


class ChunkerPort(Protocol):
    def create_chunks(self, doc: Document) -> List[Chunk]:
        ...


class EmbeddingProviderPort(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class EmbeddingServicePort(Protocol):
    def embed_chunks(self, chunks: List[Chunk]) -> List[Embedding]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class DocumentRepositoryPort(Protocol):
    def save(self, doc: Document) -> None:
        ...


class PatientRepositoryPort(Protocol):
    def upsert(self, document: Document, chunks: List[Chunk], embeddings: List[Embedding]) -> None:
        ...

    def search(self, query_vector: List[float], patient_id: str, limit: int = 5) -> List[Chunk]:
        ...

    def delete_by_patient(self, patient_id: str) -> None:
        ...


class DatasetRepositoryPort(Protocol):
    def upsert(
        self,
        dataset_id: str,
        source_path: str,
        texts: List[str],
        embeddings: List[Sequence[float]],
    ) -> None:
        ...

    def search(
        self,
        query_vector: List[float],
        dataset_id: Optional[str] = None,
        dataset_ids: Optional[Sequence[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        ...

    def delete_by_id(self, dataset_id: str) -> None:
        ...


class CloudLLMPort(Protocol):
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        ...


class MedGemmaProviderPort(Protocol):
    def generate(self, query: str, history: List[Dict[str, Any]], max_new_tokens: int = 200) -> str:
        ...

    def generate_stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        ...


class AskServicePort(Protocol):
    def ask(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 200,
    ) -> str:
        ...

    def ask_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        ...


# Backward compatibility alias.
AskAnythingServicePort = AskServicePort


class ClinicalQueryBuilderPort(Protocol):
    def build_query(self, topic: str) -> Dict[str, str]:
        ...


class DataAcquisitionPort(Protocol):
    def run_acquisition_pipeline(
        self,
        query: str,
        limit: int = 50,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[str]:
        ...


class DatasetRegistryPort(Protocol):
    def save(self, dataset: Dataset) -> None:
        ...

    def list_all(self) -> List[Dataset]:
        ...

    def delete(self, dataset_id: str) -> bool:
        ...


class RetrievalServicePort(Protocol):
    def retrieve_for_patient(
        self,
        query_text: str,
        patient_id: str,
        limit: Optional[int] = None,
    ) -> List[Chunk]:
        ...

    def retrieve_for_dataset(
        self,
        query_text: str,
        dataset_name: str,
        limit: Optional[int] = None,
    ) -> List[KnowledgeSnippet]:
        ...

    def retrieve_for_datasets(
        self,
        query_text: str,
        dataset_names: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[KnowledgeSnippet]:
        ...


class IngestionServicePort(Protocol):
    def ingest_patient_file(self, file_path: Path, patient_id: str) -> Document:
        ...

    def ingest_dataset_file(self, file_path: Path, dataset_name: str) -> Document:
        ...
