"""
app/services/llm.py

Async-capable LLM service — wraps Llama 3.1 8B via Ollama.
Runs blocking LLM calls in a thread pool so the event loop stays free.
"""

import asyncio
from functools import lru_cache

from langchain_ollama import OllamaLLM

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(
            "connecting_ollama",
            host=settings.ollama_host,
            model=settings.llm_model,
        )
        self._llm = OllamaLLM(
            base_url=settings.ollama_host,
            model=settings.llm_model,
            num_predict=settings.llm_max_tokens,
            temperature=0.2,
            top_p=0.9,
        )
        logger.info("ollama_ready", model=settings.llm_model)

    @property
    def llm(self) -> OllamaLLM:
        return self._llm

    def invoke_sync(self, prompt: str) -> str:
        """Synchronous LLM call — runs in thread pool executor."""
        logger.info("llm_invoke_start")
        response = self._llm.invoke(prompt)
        logger.info("llm_invoke_done")
        return response.strip()

    async def invoke(self, prompt: str) -> str:
        """
        Async LLM call — offloads blocking Ollama call to thread pool.
        Allows other requests to be processed while waiting for LLM response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke_sync, prompt)

    def health_check_sync(self) -> bool:
        try:
            self._llm.invoke("ping")
            return True
        except Exception as exc:
            logger.warning("ollama_health_check_failed", error=str(exc))
            return False

    async def health_check(self) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.health_check_sync)


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()