from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    """
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ Embed a list of texts (e.g., for chunks) """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """ Embed a single query string (e.g., for user questions) """
        ...
