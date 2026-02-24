import re
from datetime import date, datetime
from typing import List, Optional

from app.shared.config import DATE_PREFERRED_ORDER

RE_PATTERN = re.compile(r"\b(\d{2,4})([-/])(\d{1,2})\2(\d{2,4})\b")


def normalize_to_date(value: object) -> Optional[date]:
    if not value:
        return None

    if isinstance(value, datetime):
        return value.date()
    
    if isinstance(value, date):
        return value

    if isinstance(value, (float, int)):
        return datetime.fromtimestamp(value).date()

    if isinstance(value, str):
        # 1. Try ISO format (2023-10-25)
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass

        # 2. Try parse_dates logic on the string itself
        extracted = parse_dates(value)
        if extracted:
            return extracted[0]

    return None


def parse_dates(text: str) -> List[date]:
    """ Parse dates from text using regex and supported formats. Returns a list of valid date objects. """
    if not text:
        return []

    configs = {
        "YMD": {"y": 0, "m": 2, "d": 3},
        "DMY": {"d": 0, "m": 2, "y": 3},
        "MDY": {"m": 0, "d": 2, "y": 3},
    }

    matches = RE_PATTERN.findall(text)
    valid_dates = []

    for m in matches:
        for fmt in DATE_PREFERRED_ORDER:
            cfg = configs[fmt]
            try:
                raw_y = int(m[cfg["y"]])
                raw_m = int(m[cfg["m"]])
                raw_d = int(m[cfg["d"]])

                if raw_y < 100:
                    raw_y += 2000

                dt = datetime(raw_y, raw_m, raw_d)
                valid_dates.append(dt.date())
                break
            except (ValueError, IndexError):
                continue

    return sorted(set(valid_dates))
