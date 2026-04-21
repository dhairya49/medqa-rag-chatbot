"""
app/services/llm.py

Async-capable LLM service — Mistral API.
Mistral free tier has no TPM (tokens per minute) limit,
making it reliable for concurrent requests and live demos.

Model: mistral-small-latest
  - Fast, capable, free tier
  - No rate limits on tokens per minute
  - ~2-3s per request

Runs blocking calls in a thread pool so the event loop stays free.
Interface is identical to the Groq/Ollama versions —
no changes needed in agent.py, tools, or routes.
"""

import asyncio
from functools import lru_cache

from langchain_mistralai import ChatMistralAI

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(
            "connecting_mistral",
            model=settings.llm_model,
        )
        self._llm = ChatMistralAI(
            api_key=settings.mistral_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
        )
        logger.info("mistral_ready", model=settings.llm_model)

    @property
    def llm(self) -> ChatMistralAI:
        return self._llm

    def invoke_sync(self, prompt: str) -> str:
        """Synchronous Mistral call — runs inside thread pool executor."""
        logger.info("llm_invoke_start")
        response = self._llm.invoke(prompt)
        logger.info("llm_invoke_done")
        # ChatMistralAI returns AIMessage — extract string content
        return response.content.strip()

    async def invoke(self, prompt: str) -> str:
        """
        Async LLM call — offloads blocking Mistral HTTP call to thread pool.
        Allows other requests to be processed while waiting for response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke_sync, prompt)

    def health_check_sync(self) -> bool:
        try:
            response = self._llm.invoke("ping")
            return bool(response.content)
        except Exception as exc:
            logger.warning("mistral_health_check_failed", error=str(exc))
            return False

    async def health_check(self) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.health_check_sync)


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()
