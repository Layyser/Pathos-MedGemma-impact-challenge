from __future__ import annotations

from typing import Callable, Optional

from app.persistence.dataset_registry import JsonDatasetRegistry
from app.persistence.datasets_repo import DatasetRepository
from app.persistence.document_repo import DocumentRepository
from app.persistence.patient_repo import PatientRepository
from app.services.chunking import MedicalChunkingService
from app.services.datasets.clinical_search import ClinicalQueryBuilder
from app.services.datasets.data_acquisition import ClinicalDataAcquisition
from app.services.datasets.dataset import DatasetService
from app.services.embedding import EmbeddingService
from app.services.ingestion import IngestionService
from app.services.parser import FileParsingService
from app.services.retrieval import RetrievalService


class ServiceContainer:
    """
    Centralizes dependency wiring so UI/API layers can construct services once.
    """

    def __init__(
        self,
        parser: Optional[FileParsingService] = None,
        chunker: Optional[MedicalChunkingService] = None,
        embedding_svc: Optional[EmbeddingService] = None,
        doc_meta_repo: Optional[DocumentRepository] = None,
        patient_repo: Optional[PatientRepository] = None,
        dataset_repo: Optional[DatasetRepository] = None,
        dataset_registry: Optional[JsonDatasetRegistry] = None,
        query_builder_factory: Optional[Callable[[], ClinicalQueryBuilder]] = None,
        data_acquisition_factory: Optional[Callable[[str], ClinicalDataAcquisition]] = None,
    ):
        parser = parser or FileParsingService()
        chunker = chunker or MedicalChunkingService()
        embedding_svc = embedding_svc or EmbeddingService()
        doc_meta_repo = doc_meta_repo or DocumentRepository()
        patient_repo = patient_repo or PatientRepository()
        dataset_repo = dataset_repo or DatasetRepository()
        dataset_registry = dataset_registry or JsonDatasetRegistry()

        self.ingestion = IngestionService(
            parser=parser,
            chunker=chunker,
            embedder=embedding_svc,
            doc_meta_repo=doc_meta_repo,
            patient_repo=patient_repo,
            dataset_repo=dataset_repo,
        )
        self.retrieval = RetrievalService(
            embedding_svc=embedding_svc,
            vector_repo=patient_repo,
            truth_db=dataset_repo,
        )
        self.datasets = DatasetService(
            registry=dataset_registry,
            ingestion_svc=self.ingestion,
            query_builder_factory=query_builder_factory,
            data_acquisition_factory=data_acquisition_factory,
            dataset_repo=dataset_repo,
        )


def build_default_container() -> ServiceContainer:
    return ServiceContainer()
