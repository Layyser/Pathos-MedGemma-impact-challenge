from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    id: str
    patient_id: str
    file_name: str
    file_path: Path
    text: str
    source: str
    created_at: datetime
    last_modified_at: datetime
    manual_date: Optional[date] = None
    parsed_dates: List[date] = field(default_factory=list)
    indexed_at: datetime = field(default_factory=datetime.now)

    @property
    def effective_date(self) -> date:
        if self.manual_date:
            return self.manual_date
        
        if self.parsed_dates:
            return max(self.parsed_dates)

        return self.created_at.date()
