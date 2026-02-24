import logging
import os
from typing import Any, Dict, Iterator, List, Optional

import torch
from llama_index.core.base.llms.types import ChatMessage, MessageRole

# LlamaIndex imports
from llama_index.llms.huggingface import HuggingFaceLLM

# PEFT for LoRA
from peft import PeftModel

from app.shared.config import ADAPTERS_DIR, MEDGEMMA_MODEL_ID

from .base import BaseMedGemmaProvider

logger = logging.getLogger(__name__)

class LocalMedGemmaProvider(BaseMedGemmaProvider):
    def __init__(self):
        logger.info("Loading MedGemma via HuggingFaceLLM: %s", MEDGEMMA_MODEL_ID)

        self.llm = HuggingFaceLLM(
            model_name=MEDGEMMA_MODEL_ID,
            tokenizer_name=MEDGEMMA_MODEL_ID,
            context_window=8192,
            max_new_tokens=1024,
            device_map="cuda",
            generate_kwargs={
                "temperature": 0,
                "do_sample": False,
            },
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": True,
                "trust_remote_code": True,
            },
            # Strict Gemma Formatting: Maps roles to <start_of_turn>user/model
            messages_to_prompt=lambda messages: "".join(
                f"<start_of_turn>{'model' if m.role == MessageRole.ASSISTANT else 'user'}\n{m.content}<end_of_turn>\n"
                for m in messages
            ) + "<start_of_turn>model\n",
            completion_to_prompt=lambda prompt: (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ),
        )
        
        self._base_model = self.llm._model
        self.active_adapter_name: Optional[str] = None

    def load_adapter(self, adapter_name: str) -> None:
        """
        Dynamically loads a LoRA adapter. If an adapter is already loaded, 
        it adds the new one and switches to it.
        """
        if self.active_adapter_name == adapter_name:
            logger.info("Adapter %s already active.", adapter_name)
            return

        adapter_path = os.path.join(ADAPTERS_DIR, adapter_name)
        
        if not os.path.exists(adapter_path):
            logger.error("Adapter not found at: %s", adapter_path)
            raise FileNotFoundError(f"Adapter path {adapter_path} does not exist.")

        try:
            if isinstance(self.llm._model, PeftModel):
                logger.info("Switching adapter to: %s", adapter_name)
                self.llm._model.load_adapter(adapter_path, adapter_name=adapter_name)
                self.llm._model.set_adapter(adapter_name)
            else:
                logger.info("First time loading adapter. Wrapping base model with: %s", adapter_name)
                self.llm._model = PeftModel.from_pretrained(
                    self._base_model,
                    adapter_path,
                    adapter_name=adapter_name
                )
            
            self.active_adapter_name = adapter_name
            logger.info("Adapter %s loaded successfully.", adapter_name)

        except Exception as e:
            logger.error("Failed to load adapter %s: %s", adapter_name, e)
            raise

    def unload_adapter(self) -> None:
        """
        Unloads all adapters and reverts the LLM to the base model.
        """
        if not isinstance(self.llm._model, PeftModel):
            logger.info("No adapter is currently loaded.")
            return

        logger.info("Unloading adapter %s and reverting to base model.", self.active_adapter_name)
        
        self.llm._model = self._base_model
        self.active_adapter_name = None
        
        # Optional: Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _build_chat_messages(query: str, history: List[Dict[str, Any]]) -> List[ChatMessage]:
        chat_messages: List[ChatMessage] = []
        for h in history:
            role = MessageRole.USER if h["role"] == "user" else MessageRole.ASSISTANT
            chat_messages.append(ChatMessage(role=role, content=h["content"]))

        chat_messages.append(ChatMessage(role=MessageRole.USER, content=query))
        return chat_messages

    def generate(self, query: str, history: List[Dict[str, Any]], max_new_tokens: int = 200) -> str:
        chat_messages = self._build_chat_messages(query=query, history=history)

        response = self.llm.chat(chat_messages, max_new_tokens=max_new_tokens)

        return response.message.content or ""

    def generate_stream(
        self,
        query: str,
        history: List[Dict[str, Any]],
        max_new_tokens: int = 200,
    ) -> Iterator[str]:
        chat_messages = self._build_chat_messages(query=query, history=history)
        stream = self.llm.stream_chat(chat_messages, max_new_tokens=max_new_tokens)

        assembled = ""
        for partial in stream:
            if partial.delta:
                assembled += partial.delta
                yield partial.delta
                continue

            message_text = partial.message.content if partial.message else ""
            if not message_text:
                continue

            if message_text.startswith(assembled):
                delta = message_text[len(assembled):]
            else:
                delta = message_text

            if delta:
                yield delta
            assembled = message_text
