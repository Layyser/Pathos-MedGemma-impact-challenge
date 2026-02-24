from typing import Any, Dict, Optional, Sequence

from app.services.ports import RetrievalServicePort
from app.services.retrieval import RetrievalService


def run_patient_retrieval_pipeline(
    query_text: str,
    patient_id: str,
    limit: Optional[int] = None,
    retrieval_svc: Optional[RetrievalServicePort] = None,
) -> Dict[str, Any]:
    svc = retrieval_svc or RetrievalService()
    chunks = svc.retrieve_for_patient(query_text=query_text, patient_id=patient_id, limit=limit)

    return {
        "query": query_text,
        "patient_id": patient_id,
        "count": len(chunks),
        "results": [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            }
            for chunk in chunks
        ],
    }


def run_dataset_retrieval_pipeline(
    query_text: str,
    dataset_name: str,
    limit: Optional[int] = None,
    retrieval_svc: Optional[RetrievalServicePort] = None,
) -> Dict[str, Any]:
    svc = retrieval_svc or RetrievalService()
    snippets = svc.retrieve_for_dataset(query_text=query_text, dataset_name=dataset_name, limit=limit)

    return {
        "query": query_text,
        "dataset_name": dataset_name,
        "count": len(snippets),
        "results": [
            {
                "id": snippet.id,
                "text": snippet.text,
                "source": snippet.source,
                "score": snippet.score,
                "metadata": snippet.metadata,
            }
            for snippet in snippets
        ],
    }


def run_multi_dataset_retrieval_pipeline(
    query_text: str,
    dataset_names: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    retrieval_svc: Optional[RetrievalServicePort] = None,
) -> Dict[str, Any]:
    svc = retrieval_svc or RetrievalService()
    normalized_names = [name for name in (dataset_names or []) if name]
    snippets = svc.retrieve_for_datasets(
        query_text=query_text,
        dataset_names=normalized_names if normalized_names else None,
        limit=limit,
    )

    return {
        "query": query_text,
        "dataset_names": normalized_names,
        "count": len(snippets),
        "results": [
            {
                "id": snippet.id,
                "text": snippet.text,
                "source": snippet.source,
                "score": snippet.score,
                "metadata": snippet.metadata,
            }
            for snippet in snippets
        ],
    }
