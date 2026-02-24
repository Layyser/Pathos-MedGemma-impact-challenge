import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.ingestion import IngestionService
from app.services.ports import IngestionServicePort

logger = logging.getLogger(__name__)


def run_patient_files_ingestion_pipeline(
    file_paths: List[Path],
    patient_id: str,
    ingestion_svc: Optional[IngestionServicePort] = None,
) -> Dict[str, Any]:
    """
    Batch processes a list of patient files.
    Returns a summary of successes and failures.
    """
    logger.info("Starting batch ingestion for patient %s with %d files.", patient_id, len(file_paths))
    
    svc = ingestion_svc or IngestionService()
    
    results = []
    success_count = 0
    failure_count = 0

    for file_path in file_paths:
        try:
            # 1. Delegate the heavy lifting to the Service
            # It handles Parsing -> Chunking -> Embedding -> RAM Upsert
            document = svc.ingest_patient_file(file_path, patient_id)
            
            success_count += 1
            results.append({
                "file_name": file_path.name,
                "status": "success",
                "document_id": document.id,
                "timestamp": document.created_at.isoformat()
            })
            logger.debug("Successfully ingested: %s", file_path.name)

        except Exception as e:
            # 2. Catch individual file errors so the batch continues
            failure_count += 1
            logger.error("Failed to ingest %s: %s", file_path.name, e)
            results.append({
                "file_name": file_path.name,
                "status": "error",
                "message": str(e)
            })

    # 3. Return a UI-friendly summary
    return {
        "summary": {
            "total": len(file_paths),
            "success": success_count,
            "failed": failure_count
        },
        "details": results
    }
