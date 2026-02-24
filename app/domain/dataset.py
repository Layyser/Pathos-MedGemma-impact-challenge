from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto


class DatasetStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class Dataset:
    id: str
    name: str
    source: str
    status: DatasetStatus
    ingestion_date: datetime
    document_count: int
    raw_filter: str = ""
    topic: str = ""
