import logging
from typing import List, Optional

from app.domain.chunk import Chunk
from app.domain.embedding import Embedding
from app.providers.embeddings.factory import EmbeddingProviderFactory
from app.services.ports import EmbeddingProviderPort
from app.shared.config import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, embedder: Optional[EmbeddingProviderPort] = None):
        self.embedder = embedder or EmbeddingProviderFactory.get_provider()

    def embed_chunks(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Used during INGESTION.
        Takes a list of Chunks, batches them, and returns Embedding domain objects.
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            return []

        all_embeddings: List[Embedding] = []
        
        logger.info("Embedding %d chunks in batches of %d...", total_chunks, EMBEDDING_BATCH_SIZE)

        # Process in batches
        for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            batch_texts = [c.text for c in batch]
            
            batch_vectors = self.embedder.embed_documents(batch_texts)
            all_embeddings.extend(
                Embedding(chunk_id=chunk.id, vector=vector)
                for chunk, vector in zip(batch, batch_vectors, strict=False)
            )
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Used during RETRIEVAL.
        Takes a single query string and returns a raw vector (List[float]).
        """
        return self.embedder.embed_query(text)
