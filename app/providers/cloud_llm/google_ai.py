import logging
from typing import Optional

from google import genai
from google.api_core import exceptions

from app.shared.config import GOOGLE_API_KEY, GOOGLE_MODEL_PRIORITY_LIST

from .base import BaseCloudProvider

logger = logging.getLogger(__name__)


class GoogleAIProvider(BaseCloudProvider):
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing.")
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_list = GOOGLE_MODEL_PRIORITY_LIST

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # Initialize to avoid "possibly unbound" error
        last_exception: Optional[Exception] = None

        for model_id in self.model_list:
            try:
                logger.info(f"Attempting generation with: {model_id}")
                response = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={'temperature': temperature}
                )
                
                if response and response.text:
                    return response.text
                
                continue # If response is empty, try the next model

            except exceptions.ResourceExhausted as e:
                logger.warning(f"Rate limit hit for {model_id}. Falling back...")
                last_exception = e
                continue 
            
            except Exception as e:
                logger.error(f"Critical failure on model {model_id}: {str(e)}")
                last_exception = e
                continue

        # If we reach here, the loop finished without a successful return
        logger.error("All models in fallback chain failed.")
        raise last_exception or Exception("Fallback chain exhausted with no models tried.")