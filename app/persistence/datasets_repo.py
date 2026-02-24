import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.api.types import Metadata

from app.shared.config import CHROMA_TRUTH_COLLECTION_NAME, DATA_DIR, PERSIST_DIR

logger = logging.getLogger(__name__)


class DatasetRepository:
    _client = None
    _collection = None
    _title_line_pattern = re.compile(r"^\s*TITLE:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    _title_inline_pattern = re.compile(r"TITLE:\s*(.+?)(?:\s+ABSTRACT:|$)", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        client: Optional[Any] = None,
        collection: Optional[Any] = None,
        persist_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initializes the connection to the PERSISTENT Chroma instance.
        """
        if collection is not None:
            self.client = client
            self.collection = collection
            return

        if client is not None:
            self.client = client
            self.collection = client.get_or_create_collection(
                name=CHROMA_TRUTH_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return

        if DatasetRepository._client is None:
            store_path = Path(persist_path) if persist_path is not None else Path(PERSIST_DIR)
            store_path.mkdir(parents=True, exist_ok=True)
             
            logger.debug("Initializing persistent DatasetRepository at %s", store_path)
            DatasetRepository._client = chromadb.PersistentClient(path=str(store_path))
            
            DatasetRepository._collection = (
                DatasetRepository._client.get_or_create_collection(
                    name=CHROMA_TRUTH_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            )

        if DatasetRepository._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB ground-truth collection")

        self.client = DatasetRepository._client
        self.collection = DatasetRepository._collection

    @classmethod
    def _extract_title_from_text(cls, text: str) -> Optional[str]:
        if not text:
            return None

        line_match = cls._title_line_pattern.search(text)
        if line_match:
            title = line_match.group(1).strip()
            return title or None

        inline_match = cls._title_inline_pattern.search(text)
        if inline_match:
            title = inline_match.group(1).strip()
            return title or None

        return None

    @staticmethod
    def _candidate_source_paths(dataset_id: str, source_path: str) -> List[Path]:
        source_candidate = Path(source_path)
        paths: List[Path] = []

        if source_candidate.is_absolute():
            paths.append(source_candidate)

        data_root = Path(DATA_DIR)
        dataset_variants = [dataset_id, dataset_id.lower().replace(" ", "_")]

        for ds in dataset_variants:
            paths.append(data_root / ds / source_path)

        paths.append(data_root / source_path)

        # Preserve order while removing duplicates.
        deduped: List[Path] = []
        seen = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)

        return deduped

    @classmethod
    def _extract_title_from_source_file(cls, dataset_id: str, source_path: str) -> Optional[str]:
        if not dataset_id or not source_path:
            return None

        for path in cls._candidate_source_paths(dataset_id=dataset_id, source_path=source_path):
            if not path.exists() or not path.is_file():
                continue

            try:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            title = cls._extract_title_from_text(raw_text)
            if title:
                return title

        return None

    def upsert(
        self,
        dataset_id: str,
        source_path: str,
        texts: List[str],
        embeddings: List[Sequence[float]],
    ) -> None:
        """
        Saves truth chunks and embeddings to disk.
        """
        if not dataset_id or not texts:
            return

        ids: List[str] = []
        vectors: List[Sequence[float]] = []
        documents_text: List[str] = []
        metadatas: List[Metadata] = []
        title = self._extract_title_from_text(texts[0]) if texts else None
        if not title:
            title = self._extract_title_from_source_file(dataset_id=dataset_id, source_path=source_path)

        for idx, text in enumerate(texts):
            if idx >= len(embeddings):
                break

            # 1. Deterministic ID Generation
            # Incorporating 'idx' ensures we can have duplicate text in a doc without ID collision
            payload = f"{dataset_id}:{source_path}:{idx}:{text}"
            stable_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            ids.append(stable_id)
            documents_text.append(text)
            vectors.append(embeddings[idx])

            metadatas.append({
                "dataset_id": dataset_id,
                "source": source_path,
                "type": "ground_truth", # Useful for citing specific parts of a guideline later
                "start_offset": 0,
                "end_offset": len(text),
                "title": title or "",
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=documents_text,
                metadatas=metadatas,
            )
            logger.info(
                "Upserted %d truth chunks for dataset %s from %s",
                len(ids),
                dataset_id,
                source_path,
            )

    def search(
        self,
        query_vector: List[float],
        dataset_id: Optional[str] = None,
        dataset_ids: Optional[Sequence[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Searches the persistent database for relevant ground truth.
        """
        query_args: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        
        # Apply filters when targeting one or many dataset IDs.
        normalized_dataset_ids = [ds for ds in (dataset_ids or []) if ds]
        if normalized_dataset_ids:
            if len(normalized_dataset_ids) == 1:
                query_args["where"] = {"dataset_id": normalized_dataset_ids[0]}
            else:
                query_args["where"] = {"dataset_id": {"$in": normalized_dataset_ids}}
        elif dataset_id:
            query_args["where"] = {"dataset_id": dataset_id}

        results = self.collection.query(**query_args)

        # Chroma returns lists of lists (batch format). We only processed one query.
        raw_ids = results.get("ids")
        if not raw_ids or not raw_ids[0]:
            return []

        batch_ids = raw_ids[0]
        batch_metas = (results.get("metadatas") or [[]])[0]
        batch_texts = (results.get("documents") or [[]])[0]
        batch_dists = (results.get("distances") or [[]])[0]

        found_items: List[Dict[str, Any]] = []

        for i, item_id in enumerate(batch_ids):
            # Safety check for index out of bounds (rare but possible in some DB states)
            if i >= len(batch_metas) or i >= len(batch_texts):
                break

            meta = batch_metas[i] or {}
            text = batch_texts[i]
            if not isinstance(meta, dict):
                meta = {}
            else:
                meta = dict(meta)
            
            # Default to 1.0 distance (worst score) if missing
            distance = batch_dists[i] if batch_dists and i < len(batch_dists) else 1.0

            # 2. FLATTENING LOGIC (The Improvement)
            # Pull 'source' up to the top level so RetrievalService doesn't have to dig
            source = meta.get("source", "Unknown Source")
            dataset_name = str(meta.get("dataset_id", "")).strip()
            title = str(meta.get("title", "")).strip()
            if not title:
                title = self._extract_title_from_text(text) or ""
            if not title and dataset_name and isinstance(source, str):
                title = self._extract_title_from_source_file(dataset_id=dataset_name, source_path=source) or ""
            if title:
                meta["title"] = title

            found_items.append({
                "id": item_id,
                "text": text,
                "source": source,    # <--- Direct access for KnowledgeSnippet
                "metadata": meta,
                "title": title,
                "score": 1.0 - float(distance), # Convert Distance to Similarity
            })

        return found_items
    
    def delete_by_id(self, dataset_id: str) -> None:
        """
        Removes all documents and embeddings associated with a specific dataset_id.
        """
        if not dataset_id:
            logger.warning("Attempted to delete with empty dataset_id.")
            return

        try:
            logger.info("Deleting all entries for dataset_id: %s", dataset_id)
            
            # ChromaDB delete allows filtering by metadata using the 'where' clause
            self.collection.delete(
                where={"dataset_id": dataset_id}
            )
            
            logger.info("Successfully deleted entries for dataset: %s", dataset_id)
            
        except Exception as e:
            logger.error("Failed to delete entries for dataset %s: %s", dataset_id, str(e))
            raise e
