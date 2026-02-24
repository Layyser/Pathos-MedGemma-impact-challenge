from app.shared.config import MEDGEMMA_PROVIDER_TYPE  # "local" or "hospital_api"

from .base import BaseMedGemmaProvider
from .local import LocalMedGemmaProvider
from .remote import RemoteMedGemmaProvider


class MedGemmaFactory:
    _instance = None

    @staticmethod
    def get_provider() -> BaseMedGemmaProvider:
        if MedGemmaFactory._instance is None:
            if MEDGEMMA_PROVIDER_TYPE == "local":
                MedGemmaFactory._instance = LocalMedGemmaProvider()
            elif MEDGEMMA_PROVIDER_TYPE == "hospital_api":
                MedGemmaFactory._instance = RemoteMedGemmaProvider()
            else:
                raise ValueError(f"Unknown provider: {MEDGEMMA_PROVIDER_TYPE}")
        
        return MedGemmaFactory._instance
