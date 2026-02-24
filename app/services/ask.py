from typing import Any, Dict, Iterator, List, Optional

from app.providers.llm.factory import MedGemmaFactory
from app.services.ports import MedGemmaProviderPort


class AskService:
    """
    Minimal chat service backed by the MedGemma provider (local/hospital, not cloud).
    """

    def __init__(self, provider: Optional[MedGemmaProviderPort] = None):
        self.provider = provider or MedGemmaFactory.get_provider()

    @staticmethod
    def _normalize_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return list(history) if history else []

    def ask(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 200,
    ) -> str:
        return self.provider.generate(
            query=prompt,
            history=self._normalize_history(history),
            max_new_tokens=max_new_tokens,
        )

    def ask_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        yield from self.provider.generate_stream(
            query=prompt,
            history=self._normalize_history(history),
            max_new_tokens=max_new_tokens,
        )

