import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from app.domain.dataset import Dataset, DatasetStatus
from app.shared.config import DATASET_REGISTRY

logger = logging.getLogger(__name__)


class JsonDatasetRegistry:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or DATASET_REGISTRY
        self._storage: Dict[str, Dataset] = {}
        self._ensure_storage_directory()
        self._load_from_disk()

    @staticmethod
    def _parse_status(raw_status: object) -> DatasetStatus:
        if isinstance(raw_status, DatasetStatus):
            return raw_status

        if isinstance(raw_status, str):
            normalized = raw_status.strip()
            if normalized in DatasetStatus.__members__:
                return DatasetStatus[normalized]

            # Backward compatibility for values serialized like "DatasetStatus.COMPLETED"
            if "." in normalized:
                maybe_name = normalized.split(".")[-1]
                if maybe_name in DatasetStatus.__members__:
                    return DatasetStatus[maybe_name]

        return DatasetStatus.PENDING

    @staticmethod
    def _parse_ingestion_date(raw_date: object) -> datetime:
        if isinstance(raw_date, datetime):
            return raw_date

        if isinstance(raw_date, str):
            try:
                return datetime.fromisoformat(raw_date)
            except ValueError:
                pass

        return datetime.now()

    def _ensure_storage_directory(self):
        """Creates the directory if it doesn't exist."""
        directory = os.path.dirname(self.file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _load_from_disk(self):
        """Loads the registry from the JSON file into memory."""
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Reconstruct Domain Entities from JSON dicts
                for item in data:
                    try:
                        dataset = Dataset(
                            id=item["id"],
                            name=item["name"],
                            source=item["source"],
                            status=self._parse_status(item.get("status")),
                            ingestion_date=self._parse_ingestion_date(item.get("ingestion_date")),
                            document_count=int(item.get("document_count", 0)),
                            raw_filter=str(item.get("raw_filter", "")),
                            topic=str(item.get("topic", "")),
                        )
                        self._storage[dataset.id] = dataset
                    except (KeyError, TypeError, ValueError):
                        logger.warning("Skipping invalid dataset entry in registry: %s", item)
        except json.JSONDecodeError:
            logger.warning("Corrupted registry file at %s. Starting fresh.", self.file_path)
            self._storage = {}

    def _save_to_disk(self):
        """Dumps the current memory state to the JSON file."""
        data = []
        for dataset in self._storage.values():
            status_value = dataset.status.name if isinstance(dataset.status, DatasetStatus) else str(dataset.status)
            ingestion_value = (
                dataset.ingestion_date.isoformat()
                if isinstance(dataset.ingestion_date, datetime)
                else str(dataset.ingestion_date)
            )

            # Convert Domain Entity to Dict
            data.append({
                "id": dataset.id,
                "name": dataset.name,
                "source": dataset.source,
                "status": status_value,
                "ingestion_date": ingestion_value,
                "document_count": dataset.document_count,
                "raw_filter": dataset.raw_filter,
                "topic": dataset.topic,
            })
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    # --- Public Interface ---

    def save(self, dataset: Dataset) -> None:
        """Saves or updates a dataset and persists to disk."""
        self._storage[dataset.id] = dataset
        self._save_to_disk()

    def get_by_id(self, dataset_id: str) -> Optional[Dataset]:
        return self._storage.get(dataset_id)

    def get_by_name(self, name: str) -> Optional[Dataset]:
        for dataset in self._storage.values():
            if dataset.name == name:
                return dataset
        return None

    def list_all(self) -> List[Dataset]:
        return list(self._storage.values())

    def delete(self, dataset_id: str) -> bool:
        if dataset_id in self._storage:
            del self._storage[dataset_id]
            self._save_to_disk()
            return True
        return False
