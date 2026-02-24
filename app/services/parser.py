# app/services/file_parsing_service.py

import logging
from pathlib import Path
from typing import Callable, Dict

import docx
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class FileParsingService:
    """
    Purely responsible for turning a file path into a string string.
    No database logic here.
    """
    def __init__(self):
        self._extractors: Dict[str, Callable[[Path], str]] = {
            '.txt': self._read_txt,
            '.md': self._read_txt,
            '.pdf': self._read_pdf,
            '.docx': self._read_docx
        }

    def parse_text(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        extractor = self._extractors.get(suffix)

        if not extractor:
            raise ValueError(f"Unsupported file format: {suffix}")

        try:
            text = extractor(file_path)
            if not text.strip():
                logger.warning(f"File {file_path.name} is empty.")
            return text
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {e}")
            raise e

    def _read_txt(self, p: Path) -> str:
        return p.read_text(encoding="utf-8", errors="replace")

    def _read_pdf(self, p: Path) -> str:
        reader = PdfReader(str(p))
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def _read_docx(self, p: Path) -> str:
        doc = docx.Document(str(p))
        return "\n".join([para.text for para in doc.paragraphs])