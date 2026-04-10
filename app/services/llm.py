"""
app/services/llm.py

Async-capable LLM service — calls Llama 3.1 8B via Ollama REST API directly.

Why direct httpx instead of langchain_ollama:
  - langchain_ollama does not reliably pass num_predict to Ollama
  - Direct API call confirmed working via curl (done_reason=length, eval_count=50)
  - Removes unnecessary abstraction layer for a simple generate call

Two token limits:
  - concise  mode → 150 tokens  (~8-10s response time)
  - detailed mode → llm_max_tokens from config (full response)
"""

import asyncio
import httpx
from functools import lru_cache

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CONCISE_TOKENS = 150


class LLMService:

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._generate_url = f"{settings.ollama_host}/api/generate"
        logger.info(
            "llm_service_ready",
            host=settings.ollama_host,
            model=settings.llm_model,
        )

    def invoke_sync(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Synchronous Ollama call via httpx — runs in thread pool executor.
        Calls /api/generate directly so num_predict is always respected.
        """
        tokens = max_tokens or self._settings.llm_max_tokens  # ← add this
        logger.info("llm_invoke_start", max_tokens=max_tokens)
        payload = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": tokens,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        response = httpx.post(
            self._generate_url,
            json=payload,
            timeout=180.0,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()
        logger.info("llm_invoke_done")
        return answer

    async def invoke(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Async LLM call — offloads blocking httpx call to thread pool.
        Allows other requests to be processed while waiting for response.

        Args:
            prompt     : fully formatted prompt string
            max_tokens : hard token limit — pass 150 for concise, None for detailed
        """
        tokens = max_tokens or self._settings.llm_max_tokens
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.invoke_sync, prompt, max_tokens
        )

    async def health_check(self) -> bool:
        """Ping Ollama with a minimal request. Used by GET /health."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.invoke_sync, "ping", 5
            )
            return True
        except Exception as exc:
            logger.warning("ollama_health_check_failed", error=str(exc))
            return False


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()