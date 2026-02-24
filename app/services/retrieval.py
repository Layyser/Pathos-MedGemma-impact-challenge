from typing import Any, Dict, List, Optional, Sequence

from app.domain.chunk import Chunk
from app.domain.knowledge import KnowledgeSnippet
from app.persistence.datasets_repo import DatasetRepository
from app.persistence.patient_repo import PatientRepository
from app.services.embedding import EmbeddingService
from app.services.ports import (
    DatasetRepositoryPort,
    EmbeddingServicePort,
    PatientRepositoryPort,
)
from app.shared.config import TOP_K_RETRIEVAL


class RetrievalService:
    """
    Retrieves patient chunks from ephemeral vector storage and
    dataset ground-truth snippets from persistent storage.
    """

    def __init__(
        self,
        embedding_svc: Optional[EmbeddingServicePort] = None,
        vector_repo: Optional[PatientRepositoryPort] = None,
        truth_db: Optional[DatasetRepositoryPort] = None,
    ):
        self.embedding_svc = embedding_svc or EmbeddingService()
        self.vector_repo = vector_repo or PatientRepository()
        self.truth_db = truth_db or DatasetRepository()

    def retrieve_for_patient(
        self,
        query_text: str,
        patient_id: str,
        limit: Optional[int] = None,
    ) -> List[Chunk]:
        query_vector = self.embedding_svc.embed_query(query_text)
        top_k = limit if limit and limit > 0 else TOP_K_RETRIEVAL
        return self.vector_repo.search(query_vector, patient_id, limit=top_k)

    def retrieve_for_dataset(
        self,
        query_text: str,
        dataset_name: str,
        limit: Optional[int] = None,
    ) -> List[KnowledgeSnippet]:
        return self.retrieve_for_datasets(
            query_text=query_text,
            dataset_names=[dataset_name],
            limit=limit,
        )

    def retrieve_for_datasets(
        self,
        query_text: str,
        dataset_names: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[KnowledgeSnippet]:
        query_vector = self.embedding_svc.embed_query(query_text)
        top_k = limit if limit and limit > 0 else TOP_K_RETRIEVAL
        normalized_names = [name for name in (dataset_names or []) if name]

        truth_hits = self.truth_db.search(
            query_vector=query_vector,
            dataset_id=None,
            dataset_ids=normalized_names if normalized_names else None,
            limit=top_k,
        )

        return self._to_knowledge_snippets(truth_hits)

    def retrieve_for_dataset_id(
        self,
        query_text: str,
        dataset_id: str,
        limit: Optional[int] = None,
    ) -> List[KnowledgeSnippet]:
        return self.retrieve_for_dataset(
            query_text=query_text,
            dataset_name=dataset_id,
            limit=limit,
        )

    @staticmethod
    def _to_knowledge_snippets(truth_hits: List[Dict[str, Any]]) -> List[KnowledgeSnippet]:
        results: List[KnowledgeSnippet] = []
        for hit in truth_hits:
            if not hit.get("text"):
                continue

            results.append(
                KnowledgeSnippet(
                    text=hit["text"],
                    source=hit.get("source", "Unknown Source"),
                    score=hit.get("score", 0.0),
                    id=hit.get("id"),
                    metadata=hit.get("metadata", {}),
                )
            )

        return results
