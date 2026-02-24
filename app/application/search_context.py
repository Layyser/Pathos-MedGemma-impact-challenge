import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.application.container import ServiceContainer
from app.shared.config import TOP_K_RETRIEVAL

_INTERSECTION_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_INTERSECTION_STOPWORDS = {
    "and", "are", "but", "for", "from", "has", "have", "into", "not", "the",
    "their", "them", "there", "these", "they", "this", "was", "were", "with",
}


def _patient_ids_from_uploaded_docs(container: ServiceContainer) -> List[str]:
    # Keep this simple: in-memory document repo tracks indexed patient IDs.
    return sorted(container.ingestion.doc_meta_repo._patient_docs.keys())


def _normalize_selected_datasets(
    selected_datasets: Optional[Sequence[str]] = None,
    selected_dataset: Optional[str] = None,
) -> List[str]:
    normalized: List[str] = []
    for value in selected_datasets or []:
        if value and value not in normalized:
            normalized.append(value)

    if selected_dataset and selected_dataset not in normalized:
        normalized.append(selected_dataset)

    return normalized


def _normalize_selected_document_ids(
    selected_document_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    normalized: List[str] = []
    for value in selected_document_ids or []:
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _search_selected_docs(
    container: ServiceContainer,
    selected_document_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    normalized_ids = _normalize_selected_document_ids(selected_document_ids)
    if not normalized_ids:
        return []

    doc_repo = container.ingestion.doc_meta_repo
    selected_docs = doc_repo.fetch_by_ids(normalized_ids)
    selected_docs.sort(
        key=lambda doc: (
            doc.effective_date.isoformat(),
            doc.patient_id,
            doc.file_name.lower(),
            doc.id,
        )
    )

    return [
        {
            "id": doc.id,
            "document_id": doc.id,
            "patient_id": doc.patient_id,
            "source": doc.file_name,
            "text": doc.text,
            "start_offset": 0,
            "end_offset": len(doc.text),
            "effective_date": doc.effective_date.isoformat(),
            "score": None,
        }
        for doc in selected_docs
        if doc.text
    ]


def _search_patient_docs_similarity(
    container: ServiceContainer,
    query: str,
    limit: int,
    score_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    top_k = max(1, limit)
    doc_repo = container.ingestion.doc_meta_repo
    doc_hits: List[Dict[str, Any]] = []

    for patient_id in _patient_ids_from_uploaded_docs(container):
        chunks = container.retrieval.retrieve_for_patient(
            query_text=query,
            patient_id=patient_id,
            limit=top_k,
        )
        for chunk in chunks:
            source_doc = doc_repo.get_by_id(chunk.document_id)
            score = float(chunk.score) if chunk.score is not None else None
            if score is None or score < score_threshold:
                continue
            doc_hits.append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "patient_id": chunk.patient_id,
                    "source": source_doc.file_name if source_doc else "Unknown document",
                    "text": chunk.text,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "effective_date": source_doc.effective_date.isoformat() if source_doc else "",
                    "score": score,
                }
            )

    doc_hits.sort(
        key=lambda item: (
            -(float(item["score"]) if isinstance(item.get("score"), (int, float)) else -1.0),
            item.get("effective_date") or "9999-12-31",
            str(item.get("patient_id", "")),
            str(item.get("source", "")),
            int(item.get("start_offset", 0)),
        )
    )

    return doc_hits[:top_k]


def _search_dataset(
    container: ServiceContainer,
    query: str,
    selected_datasets: Optional[Sequence[str]],
    limit: int,
    score_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    normalized_selection = [name for name in (selected_datasets or []) if name]
    snippets = container.retrieval.retrieve_for_datasets(
        query_text=query,
        dataset_names=normalized_selection if normalized_selection else None,
        limit=limit,
    )

    return [
        {
            "id": snippet.id,
            "source": snippet.source,
            "score": snippet.score,
            "text": snippet.text,
            "metadata": snippet.metadata,
        }
        for snippet in snippets
        if snippet.score >= score_threshold
    ]


def _content_tokens(text: str) -> Set[str]:
    tokens = _INTERSECTION_TOKEN_RE.findall(text.lower())
    return {
        token
        for token in tokens
        if token not in _INTERSECTION_STOPWORDS and not token.isdigit()
    }


def _intersect_patient_and_dataset(
    docs: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not docs or not dataset:
        return [], []

    doc_token_sets = [_content_tokens(str(item.get("text", ""))) for item in docs]
    dataset_token_sets = [_content_tokens(str(item.get("text", ""))) for item in dataset]

    keep_docs_idx = [
        idx
        for idx, token_set in enumerate(doc_token_sets)
        if token_set and any(token_set.intersection(ds_tokens) for ds_tokens in dataset_token_sets)
    ]
    keep_dataset_idx = [
        idx
        for idx, token_set in enumerate(dataset_token_sets)
        if token_set and any(token_set.intersection(doc_tokens) for doc_tokens in doc_token_sets)
    ]

    return [docs[idx] for idx in keep_docs_idx], [dataset[idx] for idx in keep_dataset_idx]


def run_search(
    container: ServiceContainer,
    scope: str,
    query: str,
    selected_datasets: Optional[Sequence[str]] = None,
    selected_dataset: Optional[str] = None,
    limit: Optional[int] = None,
    selected_document_ids: Optional[Sequence[str]] = None,
    score_threshold: float = 0.4,
) -> Dict[str, Any]:
    search_limit = limit if limit is not None else TOP_K_RETRIEVAL
    normalized_scope = scope.lower()
    normalized_dataset_selection = _normalize_selected_datasets(
        selected_datasets=selected_datasets,
        selected_dataset=selected_dataset,
    )
    normalized_doc_selection = _normalize_selected_document_ids(
        selected_document_ids=selected_document_ids,
    )

    docs: List[Dict[str, Any]] = []
    dataset: List[Dict[str, Any]] = []

    if normalized_scope == "selected_docs":
        docs = _search_selected_docs(
            container=container,
            selected_document_ids=normalized_doc_selection,
        )
    elif normalized_scope in {"docs", "all"}:
        docs = _search_patient_docs_similarity(
            container=container,
            query=query,
            limit=search_limit,
            score_threshold=score_threshold,
        )

    if normalized_scope in {"dataset", "all"}:
        dataset = _search_dataset(
            container=container,
            query=query,
            selected_datasets=normalized_dataset_selection,
            limit=search_limit,
            score_threshold=score_threshold,
        )

    return {
        "query": query,
        "scope": normalized_scope,
        "selected_dataset": normalized_dataset_selection[0] if len(normalized_dataset_selection) == 1 else None,
        "selected_datasets": normalized_dataset_selection,
        "selected_document_ids": normalized_doc_selection,
        "score_threshold": score_threshold,
        "docs": docs,
        "dataset": dataset,
    }
