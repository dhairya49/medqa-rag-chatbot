"""
app/api/dependencies.py

Shared FastAPI dependency injection functions.
These are injected into route handlers via FastAPI's Depends() system.

Keeps route files clean — no service instantiation logic inside routes.
All heavy singletons (agent, retriever, llm) are initialised once at
startup and reused across requests.

Depends on:
    - app/services/agent.py      (get_agent)
    - app/services/retrieval.py  (get_retrieval_service)
    - app/services/llm.py        (get_llm_service)
    - app/services/embedding.py  (get_embedding_service)
"""

from app.services.agent import get_agent, RAGAgent
from app.services.retrieval import get_retrieval_service, RetrievalService
from app.services.llm import get_llm_service, LLMService
from app.services.embedding import get_embedding_service, EmbeddingService


def get_agent_dep() -> RAGAgent:
    """
    Dependency that provides the singleton RAGAgent.

    Usage in a route:
        @router.post("/chat")
        async def chat(agent: RAGAgent = Depends(get_agent_dep)):
            ...

    Note: routes/chat.py currently calls get_agent() directly for simplicity.
    This dependency exists for explicit injection in future routes or tests
    where you want to mock the agent cleanly.
    """
    return get_agent()


def get_retrieval_dep() -> RetrievalService:
    """
    Dependency that provides the singleton RetrievalService.
    Useful for routes that need direct Qdrant access
    without going through the full agent (e.g. a /search debug endpoint).
    """
    return get_retrieval_service()


def get_llm_dep() -> LLMService:
    """
    Dependency that provides the singleton LLMService.
    """
    return get_llm_service()


def get_embedding_dep() -> EmbeddingService:
    """
    Dependency that provides the singleton EmbeddingService.
    """
    return get_embedding_service()