import logging
from typing import Any, List, Optional, Sequence

import chromadb
from chromadb.api.types import Metadata

from app.domain.chunk import Chunk
from app.domain.document import Document
from app.domain.embedding import Embedding
from app.shared.config import CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)


def require_str(v: object, name: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{name} must be str, got {type(v)}")
    return v


def require_int(v: object, name: str) -> int:
    if not isinstance(v, int):
        raise ValueError(f"{name} must be int, got {type(v)}")
    return v


class PatientRepository:
    _client = None
    _collection = None

    def __init__(
        self,
        client: Optional[Any] = None,
        collection: Optional[Any] = None,
    ):
        """
        Initializes the connection to the SHARED in-memory Chroma instance.
        """
        if collection is not None:
            self.client = client
            self.collection = collection
            return

        if client is not None:
            self.client = client
            self.collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return

        if PatientRepository._client is None:
            logger.debug("Initializing Ephemeral (RAM) PatientRepository...")
            PatientRepository._client = chromadb.EphemeralClient()
            PatientRepository._collection = PatientRepository._client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        if PatientRepository._collection is None:
            raise RuntimeError("Failed to initialize ChromaDB Collection")
            
        self.client = PatientRepository._client
        self.collection = PatientRepository._collection

    def upsert(self, document: Document, chunks: List[Chunk], embeddings: List[Embedding]) -> None:
        """
        Saves chunks and embeddings to RAM.
        """
        if not chunks:
            return

        if not document.patient_id:
            raise ValueError(f"Cannot index Document {document.id}: Missing patient_id")

        embedding_map = {e.chunk_id: e.vector for e in embeddings}

        ids: List[str] = []
        vectors: List[Sequence[float]] = []
        documents_text: List[str] = []
        metadatas: List[Metadata] = []

        for chunk in chunks:
            if chunk.patient_id != document.patient_id:
                logger.error("Patient ID mismatch in chunk %s", chunk.id)
                continue

            if chunk.id not in embedding_map:
                logger.warning("No embedding found for chunk %s, skipping.", chunk.id)
                continue

            ids.append(chunk.id)
            vectors.append(embedding_map[chunk.id])
            documents_text.append(chunk.text)

            metadatas.append({
                "patient_id": document.patient_id,
                "document_id": document.id,
                "source": document.file_name,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            })

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=documents_text,
                metadatas=metadatas,
            )
            logger.info("Upserted %d chunks for doc %s into RAM.", len(ids), document.id)

    def search(self, query_vector: List[float], patient_id: str, limit: int = 5) -> List[Chunk]:
        """
        Searches the RAM database for relevant chunks.
        """
        if not patient_id:
            raise ValueError("Search failed: patient_id is required for security.")

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={"patient_id": patient_id},
            include=["documents", "metadatas", "distances"],
        )

        raw_ids = results.get("ids") or [[]]
        raw_metas = results.get("metadatas") or [[]]
        raw_texts = results.get("documents") or [[]]
        raw_dists = results.get("distances") or [[]]

        if not raw_ids or not raw_ids[0]:
            return []

        batch_ids = raw_ids[0]
        batch_metas = raw_metas[0]
        batch_texts = raw_texts[0]
        batch_dists = raw_dists[0]

        found_chunks: List[Chunk] = []
        
        for i, doc_id in enumerate(batch_ids):
            if i >= len(batch_metas) or i >= len(batch_texts):
                break
                
            meta = batch_metas[i]
            if meta is None:
                continue

            distance = batch_dists[i] if i < len(batch_dists) else 1.0
            chunk = Chunk(
                id=doc_id,
                document_id=require_str(meta.get("document_id"), "document_id"),
                patient_id=require_str(meta.get("patient_id"), "patient_id"),
                text=batch_texts[i],
                start_offset=require_int(meta.get("start_offset"), "start_offset"),
                end_offset=require_int(meta.get("end_offset"), "end_offset"),
                score=1.0 - float(distance),
            )
            found_chunks.append(chunk)

        return found_chunks

    def delete_by_patient(self, patient_id: str) -> None:
        if not patient_id:
            raise ValueError("Delete failed: patient_id is required.")
        
        logger.info("Deleting all RAM data for patient %s", patient_id)
        self.collection.delete(where={"patient_id": patient_id})

    def delete_all(self) -> None:
        """Wipes EVERYTHING."""
        if self.client:
            self.client.reset()
