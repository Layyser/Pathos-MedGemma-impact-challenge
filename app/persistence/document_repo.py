from datetime import date
from typing import Dict, List, Optional, Set

from app.domain.document import Document


class DocumentRepository:
    _in_memory_store: Dict[str, Document] = {}
    _patient_docs: Dict[str, Set[str]] = {}

    def __init__(
        self,
        in_memory_store: Optional[Dict[str, Document]] = None,
        patient_docs: Optional[Dict[str, Set[str]]] = None,
    ):
        self._in_memory_store = in_memory_store if in_memory_store is not None else DocumentRepository._in_memory_store
        self._patient_docs = patient_docs if patient_docs is not None else DocumentRepository._patient_docs

    def save(self, doc: Document):
        """ Save a document in memory and index by patient """
        self._in_memory_store[doc.id] = doc
        self._patient_docs.setdefault(doc.patient_id, set()).add(doc.id)

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._in_memory_store.get(doc_id)

    def fetch_by_ids(self, doc_ids: List[str]) -> List[Document]:
        return [doc for doc_id in doc_ids if (doc := self.get_by_id(doc_id))]

    def fetch_by_patient_id(self, patient_id: str) -> List[Document]:
        return [doc for doc_id in self._patient_docs.get(patient_id, set()) if (doc := self.get_by_id(doc_id))]

    def set_manual_date(self, doc_id: str, manual_date: date) -> Optional[Document]:
        doc = self._in_memory_store.get(doc_id)
        if doc:
            doc.manual_date = manual_date
        return doc
