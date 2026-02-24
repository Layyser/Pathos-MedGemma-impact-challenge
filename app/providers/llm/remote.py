from typing import Any, Dict, Iterator, List

from .base import BaseMedGemmaProvider


class RemoteMedGemmaProvider(BaseMedGemmaProvider):
    def generate(self, query: str, history: List[Dict[str, Any]], max_new_tokens: int = 200) -> str:
        # TODO: Implement API logic to send images/text
        raise NotImplementedError("Remote API for MedGemma not implemented yet.")

    def load_adapter(self, adapter_name: str) -> None:
        # TODO: Implement this
        raise NotImplementedError("Remote API for MedGemma not implemented yet.")
    
    def unload_adapter(self) -> None:
        # TODO: Implement this
        raise NotImplementedError("Remote API for MedGemma not implemented yet.")

    def generate_stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        # TODO: Implement this
        raise NotImplementedError("Remote API for MedGemma not implemented yet.")
