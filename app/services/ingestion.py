import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Domain
from app.domain.document import Document
from app.persistence.datasets_repo import DatasetRepository  # DISK (Medical Knowledge)

# Repositories
from app.persistence.document_repo import (
    DocumentRepository,  # For UI lists (SQL/Metadata)
)
from app.persistence.patient_repo import PatientRepository  # RAM (Patient Data)
from app.services.chunking import MedicalChunkingService
from app.services.embedding import EmbeddingService

# Services
from app.services.parser import FileParsingService
from app.services.ports import (
    ChunkerPort,
    DatasetRepositoryPort,
    DocumentRepositoryPort,
    EmbeddingServicePort,
    ParserPort,
    PatientRepositoryPort,
)
from app.shared.date_parser import parse_dates
from app.shared.ids import generate_document_id

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the ingestion pipeline.
    Routes sensitive data to Ephemeral RAM and knowledge data to Persistent Disk.
    """

    def __init__(
        self,
        parser: Optional[ParserPort] = None,
        chunker: Optional[ChunkerPort] = None,
        embedder: Optional[EmbeddingServicePort] = None,
        doc_meta_repo: Optional[DocumentRepositoryPort] = None,
        patient_repo: Optional[PatientRepositoryPort] = None,
        dataset_repo: Optional[DatasetRepositoryPort] = None,
    ):
        self.parser = parser or FileParsingService()
        self.chunker = chunker or MedicalChunkingService()
        self.embedder = embedder or EmbeddingService()
        
        self.doc_meta_repo = doc_meta_repo or DocumentRepository()   # Keeps track of "what files are uploaded"
        self.patient_repo = patient_repo or PatientRepository()     # The actual vector store for patients
        self.dataset_repo = dataset_repo or DatasetRepository()     # The vector store for datasets

    def ingest_patient_file(self, file_path: Path, patient_id: str) -> Document:
        document = self._create_document_object(file_path, patient_id)
        
        self._run_vector_pipeline(document, is_persistent=False)
        self.doc_meta_repo.save(document)
        
        logger.info(f"Successfully ingested patient file: {file_path.name}")
        return document

    def ingest_dataset_file(self, file_path: Path, dataset_name: str) -> Document:
        document = self._create_document_object(file_path, patient_id=f"DATASET_{dataset_name}")
        
        self._run_vector_pipeline(document, is_persistent=True, dataset_id=dataset_name)

        logger.info(f"Successfully ingested knowledge file: {file_path.name}")
        return document

    def _create_document_object(self, file_path: Path, patient_id: str) -> Document:
        raw_text = self.parser.parse_text(file_path)
        found_dates = parse_dates(raw_text)
        
        return Document(
            id=generate_document_id(raw_text, patient_id),
            file_name=file_path.name,
            patient_id=patient_id,
            file_path=file_path,
            text=raw_text,
            source=file_path.suffix.lower().lstrip('.'),
            created_at=datetime.fromtimestamp(file_path.stat().st_birthtime),
            last_modified_at=datetime.fromtimestamp(file_path.stat().st_mtime),
            parsed_dates=found_dates
        )

    def _run_vector_pipeline(self, document: Document, is_persistent: bool, dataset_id: Optional[str] = None):
        """
        The core AI processing loop.
        """
        # A. Chunking
        chunks = self.chunker.create_chunks(document)
        if not chunks:
            logger.warning(f"No chunks created for {document.file_name}")
            return

        # B. Embedding
        embeddings = self.embedder.embed_chunks(chunks)
        
        # C. Storage
        if is_persistent and dataset_id is not None:
            # Route to Dataset Repo
            self.dataset_repo.upsert(
                dataset_id=dataset_id,
                source_path=document.file_name,
                texts=[c.text for c in chunks],
                embeddings=[e.vector for e in embeddings]
            )
        elif not is_persistent:
            # Route to RAM Repo
            self.patient_repo.upsert(document, chunks, embeddings)
        else:
            logger.warning("Triggered an impossible configuration")
            return
