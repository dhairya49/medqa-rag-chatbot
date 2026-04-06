"""
app/services/llm.py

Async-capable LLM service — wraps Llama 3.1 8B via Groq API.
Groq runs Llama 3.1 8B at ~500 tokens/sec vs ~20 tokens/sec locally.
Single request: ~2-3s instead of 30-120s on local hardware.

Runs blocking Groq calls in a thread pool so the event loop stays free.
Structure is identical to the Ollama version — no changes needed
in agent.py, tools, or routes.
"""

import asyncio
from functools import lru_cache

from langchain_groq import ChatGroq

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(
            "connecting_groq",
            model=settings.llm_model,
        )
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=0.2,
        )
        logger.info("groq_ready", model=settings.llm_model)

    @property
    def llm(self) -> ChatGroq:
        return self._llm

    def invoke_sync(self, prompt: str) -> str:
        """
        Synchronous Groq call — runs inside thread pool executor.
        ChatGroq returns an AIMessage object — we extract .content from it.
        """
        logger.info("llm_invoke_start")
        response = self._llm.invoke(prompt)
        logger.info("llm_invoke_done")
        # ChatGroq returns AIMessage — extract string content
        return response.content.strip()

    async def invoke(self, prompt: str) -> str:
        """
        Async LLM call — offloads blocking Groq HTTP call to thread pool.
        Allows other requests to be processed while waiting for response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke_sync, prompt)

    def health_check_sync(self) -> bool:
        try:
            response = self._llm.invoke("ping")
            return bool(response.content)
        except Exception as exc:
            logger.warning("groq_health_check_failed", error=str(exc))
            return False

    async def health_check(self) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.health_check_sync)


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()




