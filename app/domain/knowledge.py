from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KnowledgeSnippet:
    text: str
    source: str
    score: float
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)