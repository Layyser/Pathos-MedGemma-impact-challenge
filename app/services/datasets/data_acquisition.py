import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from Bio import Entrez
from bs4 import BeautifulSoup

from app.shared.config import DATA_DIR, EMAIL, RAW_DATA_DIR

logger = logging.getLogger(__name__)
Entrez.email = EMAIL


class ClinicalDataAcquisition:
    def __init__(self, dataset_name: str):
        self.dataset_slug = dataset_name.lower().replace(" ", "_")
        # Using abspath to ensure the returned paths are always valid from any directory
        self.raw_dir = os.path.abspath(os.path.join(RAW_DATA_DIR, self.dataset_slug))
        self.clean_dir = os.path.abspath(os.path.join(DATA_DIR, self.dataset_slug))
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.raw_dir, self.clean_dir]:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _emit_progress(
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        phase: str,
        message: str,
        progress: float,
        current: int = 0,
        total: int = 0,
    ) -> None:
        if progress_callback is None:
            return

        event = {
            "phase": phase,
            "message": message,
            "progress": max(0.0, min(1.0, float(progress))),
            "current": current,
            "total": total,
        }

        try:
            progress_callback(event)
        except Exception:
            logger.exception("Progress callback failed during acquisition phase.")

    def run_acquisition_pipeline(
        self,
        query: str,
        limit: int = 300,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[str]:
        """
        Executes search -> fetch -> clean.
        Returns: List of absolute paths to the cleaned .txt files.
        """
        self._emit_progress(
            progress_callback=progress_callback,
            phase="acquisition.search",
            message="Searching PubMed Central IDs...",
            progress=0.0,
        )

        pmc_ids = self._search_pmc_ids(query, limit)
        if not pmc_ids:
            logger.warning(f"No results found for {query}")
            self._emit_progress(
                progress_callback=progress_callback,
                phase="acquisition.search",
                message="No articles found for this topic.",
                progress=1.0,
            )
            return []

        processed_files = []
        total = len(pmc_ids)
        self._emit_progress(
            progress_callback=progress_callback,
            phase="acquisition.fetch",
            message=f"Found {total} articles. Fetching and cleaning...",
            progress=0.0,
            current=0,
            total=total,
        )
        
        for i, pmc_id in enumerate(pmc_ids, start=1):
            self._emit_progress(
                progress_callback=progress_callback,
                phase="acquisition.fetch",
                message=f"Fetching PMC{pmc_id} ({i}/{total})...",
                progress=(i - 1) / total,
                current=i - 1,
                total=total,
            )

            # NCBI Rate limiting (3 req/sec)
            time.sleep(0.34) 
            
            logger.info(f"[{i}/{total}] Fetching PMC{pmc_id}")
            xml_data = self._fetch_full_text(pmc_id)
            
            if xml_data:
                # 1. Save Raw XML (Optional for RAG, but good for auditing)
                raw_path = os.path.join(self.raw_dir, f"PMC{pmc_id}.xml")
                self._save_file(raw_path, xml_data, mode="wb")

                # 2. Process to Clean Text
                clean_text = self._clean_xml_to_text(xml_data)
                
                # Validation: Ensure it's not just a title/header (RAG needs substance)
                if len(clean_text) > 500: 
                    clean_path = os.path.join(self.clean_dir, f"PMC{pmc_id}.txt")
                    
                    # 3. Only append to list if the file system actually saved it
                    success = self._save_file(clean_path, clean_text, mode="w")
                    if success:
                        processed_files.append(clean_path)
                else:
                    logger.debug(f"Skipping PMC{pmc_id}: Insufficient text content.")

            self._emit_progress(
                progress_callback=progress_callback,
                phase="acquisition.fetch",
                message=f"Processed {i}/{total} articles. Kept {len(processed_files)} clean files.",
                progress=i / total,
                current=i,
                total=total,
            )
        
        logger.info(f"Pipeline complete. {len(processed_files)} documents ready.")
        self._emit_progress(
            progress_callback=progress_callback,
            phase="acquisition.complete",
            message=f"Acquisition complete. {len(processed_files)} clean documents ready.",
            progress=1.0,
            current=len(processed_files),
            total=total,
        )
        return processed_files

    def _search_pmc_ids(self, term: str, retmax: int) -> List[str]:
        try:
            with Entrez.esearch(db="pmc", term=term, retmax=retmax) as handle:
                record = Entrez.read(handle)
            return record.get("IdList", []) # type: ignore
        except Exception as e:
            logger.error(f"Entrez Search Error: {e}")
            return []

    def _fetch_full_text(self, pmc_id: str) -> Optional[bytes]:
        try:
            with Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml") as handle:
                return handle.read() # type: ignore
        except Exception as e:
            logger.error(f"Fetch Error PMC{pmc_id}: {e}")
            return None

    def _clean_xml_to_text(self, xml_content: bytes) -> str:
        """Standardizes medical text extraction and removes XML noise."""
        try:
            # Note: For massive datasets, installing/using "lxml" instead of "xml" is much faster
            soup = BeautifulSoup(xml_content, "lxml-xml") 
            
            # 1. DESTROY NOISE: Remove citations, tables, figures, and formulas
            # Doing this first prevents "fused text" and RAG hallucination triggers
            for noise in soup.find_all(["xref", "table-wrap", "fig", "disp-formula", "ext-link"]):
                noise.decompose()

            text_parts = []

            # 2. Extract Meta Information
            title = soup.find("article-title")
            if title:
                text_parts.append(f"TITLE: {title.get_text(separator=' ', strip=True)}")

            abstract = soup.find("abstract")
            if abstract:
                text_parts.append(f"ABSTRACT: {abstract.get_text(separator=' ', strip=True)}")

            # 3. Extract Body Sequentially (Solves the nested <sec> problem)
            body = soup.find("body")
            if body:
                # Iterate through titles and paragraphs in the exact order they appear
                for element in body.find_all(["title", "p"]): # type: ignore
                    text = element.get_text(separator=" ", strip=True)
                    if not text:
                        continue
                    
                    if element.name == "title": # type: ignore
                        text_parts.append(f"\n## {text}")
                    elif element.name == "p": # type: ignore
                        text_parts.append(text)

            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"XML Parsing Error: {e}")
            return ""

    def _save_file(self, path: str, content, mode: str) -> bool:
        """Saves file and returns True if successful, False otherwise."""
        try:
            if "b" in mode:
                with open(path, mode) as f:
                    f.write(content)
            else:
                with open(path, mode, encoding="utf-8") as f:
                    f.write(content)
            return True
        except Exception as e:
            logger.error(f"File Save Error {path}: {e}")
            return False
