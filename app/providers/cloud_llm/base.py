from abc import ABC, abstractmethod


class BaseCloudProvider(ABC):
    """
    Abstract base class for Cloud LLMs (e.g., Google AI Studio, OpenAI).
    Simplified for single-turn query generation tasks.
    """
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """ 
        Generates a response for a given text prompt.
        """
        ...