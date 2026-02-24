import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.shared.config import EMBEDDING_MODEL_NAME

from .base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(BaseEmbeddingProvider):
    def __init__(self):
        logger.info("Loading local Embedding model %s...", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME) 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()