from dataclasses import dataclass
from typing import List


@dataclass
class Embedding:
    chunk_id: str
    vector: List[float]