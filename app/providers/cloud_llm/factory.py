# from .openai import OpenAIProvider  <-- Easy to extend later
from app.shared.config import CLOUD_PROVIDER_TYPE

from .base import BaseCloudProvider
from .google_ai import GoogleAIProvider


class CloudLLMFactory:
    _instance = None

    @staticmethod
    def get_provider() -> BaseCloudProvider:
        if CloudLLMFactory._instance is None:
            if CLOUD_PROVIDER_TYPE == "google":
                CloudLLMFactory._instance = GoogleAIProvider()
            # elif CLOUD_PROVIDER_TYPE == "openai":
            #     CloudLLMFactory._instance = OpenAIProvider()
            else:
                raise ValueError(f"Unknown cloud provider: {CLOUD_PROVIDER_TYPE}")
        
        return CloudLLMFactory._instance