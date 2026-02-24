from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str
    document_id: str
    patient_id: str
    text: str
    start_offset: int
    end_offset: int
    score: Optional[float] = None
