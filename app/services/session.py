"""
app/services/session.py

Redis-backed session memory for multi-turn conversation history.

Each session is stored as a Redis List at key:
    session:{session_id}:history

Each list item is a JSON-serialised message:
    {"role": "user" | "assistant", "content": "..."}

TTL is sliding — reset to session_ttl_seconds on every append so active
conversations never expire mid-session. Abandoned sessions clean themselves
up automatically after 24 hours.

Usage:
    from app.services.session import SessionMemory
    memory = SessionMemory()                        # one instance, reuse it
    history = await memory.get_history("abc-123")
    await memory.append_turn("abc-123", user_msg="...", assistant_msg="...")
    await memory.clear_session("abc-123")
"""

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from app.utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Each stored message is a plain dict: {"role": str, "content": str}
Message = dict[str, str]


class SessionMemory:
    """
    Async Redis client for reading and writing per-session chat history.

    Designed to be instantiated once (e.g. at app startup via dependency
    injection) and reused across requests — the underlying connection pool
    is shared automatically by redis.asyncio.
    """

    def __init__(self) -> None:
        self._client: aioredis.Redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,      # all responses come back as str, not bytes
        )
        self._ttl: int = settings.session_ttl_seconds
        # last N turns = last N*2 messages (each turn = 1 user + 1 assistant)
        self._max_messages: int = settings.session_max_turns * 2

    # ── Key helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:history"

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[Message]:
        """
        Return the last `session_max_turns * 2` messages for this session,
        oldest first (chronological order, ready to pass directly to the LLM).

        Returns an empty list if the session does not exist yet.
        """
        key = self._key(session_id)
        try:
            # LRANGE with negative indices: -N to -1 gives last N items
            raw_messages: list[str] = await self._client.lrange(
                key, -self._max_messages, -1
            )
            return [json.loads(m) for m in raw_messages]
        except Exception:
            logger.exception("SessionMemory.get_history failed for %s", session_id)
            return []   # degrade gracefully — chat still works without history

    async def append_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """
        Append one full turn (user + assistant) to the session history and
        refresh the TTL so the session stays alive for another 24 hours.

        Both messages are pushed atomically in a single pipeline to avoid
        partial writes under concurrent load.
        """
        key = self._key(session_id)
        user_entry = json.dumps({"role": "user", "content": user_msg})
        assistant_entry = json.dumps({"role": "assistant", "content": assistant_msg})

        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, user_entry, assistant_entry)
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except Exception:
            logger.exception("SessionMemory.append_turn failed for %s", session_id)
            # Non-fatal: history just won't include this turn next time

    async def clear_session(self, session_id: str) -> None:
        """
        Delete the session history immediately.
        Called when the user clicks 'New Chat' — though the frontend also
        switches to a fresh session_id, so this is a belt-and-suspenders clean.
        """
        key = self._key(session_id)
        try:
            await self._client.delete(key)
            logger.debug("Cleared session %s", session_id)
        except Exception:
            logger.exception("SessionMemory.clear_session failed for %s", session_id)

    async def close(self) -> None:
        """
        Close the Redis connection pool. Call this on app shutdown.
        """
        await self._client.aclose()