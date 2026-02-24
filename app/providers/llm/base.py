from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class BaseMedGemmaProvider(ABC):
    """
    Abstract base class for MedGemma (Multimodal).
    """
    @abstractmethod
    def generate(self, query: str, history: List[Dict[str, Any]], max_new_tokens: int = 200) -> str:
        """ 
        Generates text based on multimodal messages (images + text).
        """
        ...

    @abstractmethod
    def generate_stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        """
        Generates text chunks as they are produced by the model.
        """
        ...

    @abstractmethod
    def load_adapter(self, adapter_name: str) -> None:
        """ 
        Loads a profile adapter
        """
        ...

    @abstractmethod
    def unload_adapter(self) -> None:
        """ 
        Unloads its currentt
        """
        ...
