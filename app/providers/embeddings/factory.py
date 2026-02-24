from app.shared.config import EMBEDDING_PROVIDER_TYPE

from .base import BaseEmbeddingProvider
from .local import SentenceTransformersProvider
from .remote import RemoteEmbeddingProvider


class EmbeddingProviderFactory:
    _instance = None

    @staticmethod
    def get_provider() -> BaseEmbeddingProvider:
        """
        Returns a singleton instance of the configured provider.
        """
        if EmbeddingProviderFactory._instance is None:
            if EMBEDDING_PROVIDER_TYPE == "local":
                EmbeddingProviderFactory._instance = SentenceTransformersProvider()
            elif EMBEDDING_PROVIDER_TYPE == "hospital_api":
                EmbeddingProviderFactory._instance = RemoteEmbeddingProvider()
            else:
                raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER_TYPE}")
        
        return EmbeddingProviderFactory._instance