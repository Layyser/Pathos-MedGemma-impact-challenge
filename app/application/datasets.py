from typing import Any, Callable, Dict, Optional

from app.application.container import ServiceContainer
from app.domain.dataset import Dataset


def get_dataset_by_name(container: ServiceContainer, dataset_name: str) -> Optional[Dataset]:
    for dataset in container.datasets.list_datasets():
        if dataset.name == dataset_name:
            return dataset
    return None


def create_dataset(
    container: ServiceContainer,
    dataset_name: str,
    topic: Optional[str] = None,
    limit: int = 300,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[Dataset]:
    dataset_topic = topic or dataset_name
    return container.datasets.create_dataset(
        name=dataset_name,
        topic=dataset_topic,
        limit=limit,
        progress_callback=progress_callback,
    )
