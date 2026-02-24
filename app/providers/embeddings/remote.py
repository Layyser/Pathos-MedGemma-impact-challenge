from typing import List

from .base import BaseEmbeddingProvider


class RemoteEmbeddingProvider(BaseEmbeddingProvider):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Remote API for Embedding models is not implemented yet.")

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Remote API for Embedding models is not implemented yet.")
