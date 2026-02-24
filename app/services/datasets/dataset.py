from datetime import datetime
from pathlib import Path
import logging
from typing import Any, Callable, Dict, List, Optional

from app.domain.dataset import Dataset, DatasetStatus
from app.persistence.datasets_repo import DatasetRepository
from app.services.datasets.clinical_search import ClinicalQueryBuilder
from app.services.datasets.data_acquisition import ClinicalDataAcquisition
from app.services.ingestion import IngestionService
from app.services.ports import (
    ClinicalQueryBuilderPort,
    DataAcquisitionPort,
    DatasetRegistryPort,
    DatasetRepositoryPort,
)

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(
        self,
        registry: DatasetRegistryPort,
        ingestion_svc: Optional[IngestionService] = None,
        query_builder_factory: Optional[Callable[[], ClinicalQueryBuilderPort]] = None,
        data_acquisition_factory: Optional[Callable[[str], DataAcquisitionPort]] = None,
        dataset_repo: Optional[DatasetRepositoryPort] = None,
    ):
        self.registry = registry
        self.ingestion_svc = ingestion_svc or IngestionService()
        self.query_builder_factory = query_builder_factory or ClinicalQueryBuilder
        self.data_acquisition_factory = data_acquisition_factory or ClinicalDataAcquisition
        self.dataset_repo = dataset_repo or DatasetRepository()

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        phase: str,
        message: str,
        progress: float,
        current: int = 0,
        total: int = 0,
    ) -> None:
        if progress_callback is None:
            return

        event = {
            "phase": phase,
            "message": message,
            "progress": max(0.0, min(1.0, float(progress))),
            "current": current,
            "total": total,
        }

        try:
            progress_callback(event)
        except Exception:
            logger.exception("Progress callback failed during dataset creation.")

    def create_dataset(
        self,
        name: str,
        topic: str,
        limit: int = 300,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Optional[Dataset]:
        self._emit_progress(
            progress_callback=progress_callback,
            phase="dataset.prepare",
            message=f"Preparing dataset '{name}'...",
            progress=0.02,
        )

        dataset = None
        dataset_id = name
        self._emit_progress(
            progress_callback=progress_callback,
            phase="dataset.query",
            message="Generating clinical search query...",
            progress=0.06,
        )
        clinical_search = self.query_builder_factory()
        built_query = clinical_search.build_query(topic)
        self._emit_progress(
            progress_callback=progress_callback,
            phase="dataset.query",
            message="Clinical query generated. Starting article acquisition...",
            progress=0.12,
        )

        files = []
        query_text = built_query.get("query")
        if query_text:
            data_acquisition = self.data_acquisition_factory(name) # Stores everything into data/name

            def acquisition_progress(event: Dict[str, Any]) -> None:
                stage_progress = float(event.get("progress", 0.0))
                overall_progress = 0.12 + (stage_progress * 0.63)
                self._emit_progress(
                    progress_callback=progress_callback,
                    phase=str(event.get("phase", "acquisition")),
                    message=str(event.get("message", "Acquiring articles...")),
                    progress=overall_progress,
                    current=int(event.get("current", 0)),
                    total=int(event.get("total", 0)),
                )

            files = data_acquisition.run_acquisition_pipeline(
                query_text,
                limit,
                progress_callback=acquisition_progress,
            )
        else:
            self._emit_progress(
                progress_callback=progress_callback,
                phase="dataset.query",
                message="Could not build a valid clinical query.",
                progress=1.0,
            )
            return None
        
        if files and built_query.get("source", None) is not None:
            dataset = Dataset(
                id=dataset_id,
                name=name,
                source=built_query["source"],
                status=DatasetStatus.COMPLETED,
                ingestion_date=datetime.now(),
                document_count=len(files),
                raw_filter=built_query.get("query", "") or "",
                topic=topic,
            )

        if not files:
            self._emit_progress(
                progress_callback=progress_callback,
                phase="dataset.acquisition",
                message="Acquisition finished, but no valid documents were generated.",
                progress=1.0,
            )
            return None

        total_files = len(files)
        for idx, file in enumerate(files, start=1):
            self._emit_progress(
                progress_callback=progress_callback,
                phase="dataset.ingestion",
                message=f"Ingesting clean document {idx}/{total_files}...",
                progress=0.78 + (idx - 1) / total_files * 0.20,
                current=idx - 1,
                total=total_files,
            )
            self.ingestion_svc.ingest_dataset_file(Path(file), name)
            self._emit_progress(
                progress_callback=progress_callback,
                phase="dataset.ingestion",
                message=f"Ingested {idx}/{total_files} documents.",
                progress=0.78 + idx / total_files * 0.20,
                current=idx,
                total=total_files,
            )

        if dataset is not None:
            self.registry.save(dataset)
            self._emit_progress(
                progress_callback=progress_callback,
                phase="dataset.complete",
                message=f"Dataset '{name}' created with {dataset.document_count} documents.",
                progress=1.0,
                current=dataset.document_count,
                total=dataset.document_count,
            )

        return dataset

    def create_dataset_from_folder(self, folder_path: Path | str, topic: Optional[str] = None) -> Optional[Dataset]:
        dataset_folder = Path(folder_path)
        if not dataset_folder.exists() or not dataset_folder.is_dir():
            return None

        dataset_id = dataset_folder.name
        dataset_name = dataset_folder.name
        dataset_topic = topic or dataset_name

        files = sorted(p for p in dataset_folder.iterdir() if p.is_file())
        if not files:
            return None

        for file in files:
            self.ingestion_svc.ingest_dataset_file(file, dataset_name)

        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            source="LocalFolder",
            status=DatasetStatus.COMPLETED,
            ingestion_date=datetime.now(),
            document_count=len(files),
            raw_filter="",
            topic=dataset_topic,
        )

        self.registry.save(dataset)
        return dataset


    def list_datasets(self) -> List[Dataset]:
        return self.registry.list_all()

    def delete_dataset(self, name: str) -> bool:
        dataset_id = name

        deleted = self.registry.delete(dataset_id)

        self.dataset_repo.delete_by_id(dataset_id)
        
        return deleted
