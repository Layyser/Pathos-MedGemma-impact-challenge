from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Patient:
    id: str
    name: Optional[str]
    birth_date: Optional[date]
    medical_record_number: Optional[str] = None

    def age(self) -> Optional[int]:
        if self.birth_date is None:
            return None
        
        today = date.today()
        return today.year - self.birth_date.year - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))
