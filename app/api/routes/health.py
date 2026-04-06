"""
app/api/routes/health.py

GET /health — liveness and readiness check endpoint.

Checks all critical runtime dependencies:
  - Qdrant       : can we connect and read the collection?
  - Ollama       : is Llama 3.1 8B reachable?

Returns overall status "ok" only if ALL dependencies are healthy.
Returns "degraded" with details if any dependency is unreachable.

Used by:
  - Docker healthcheck (docker-compose.yml)
  - Monitoring / uptime checks in Phase 4

Depends on:
  - app/services/retrieval.py  (get_retrieval_service → health_check)
  - app/services/llm.py        (get_llm_service → health_check)
  - app/models/schemas.py      (HealthResponse)
  - app/utils/config.py        (get_settings)
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.retrieval import get_retrieval_service
from app.services.llm import get_llm_service
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Dependency health check.

    Response fields:
      status       : "ok" if all deps healthy, "degraded" otherwise
      qdrant       : "connected" or "unreachable"
      ollama       : "connected" or "unreachable"
      embedding_model : name of the loaded embedding model
      collection   : Qdrant collection name
      qdrant_point_count : number of vectors stored (None if unreachable)
    """
    settings = get_settings()

    # Check Qdrant
    try:
        retriever = get_retrieval_service()
        qdrant_ok, point_count = retriever.health_check()
    except Exception as exc:
        logger.warning("health_qdrant_error", error=str(exc))
        qdrant_ok, point_count = False, 0

    # Check Ollama
    try:
        llm_service = get_llm_service()
        ollama_ok = llm_service.health_check()
    except Exception as exc:
        logger.warning("health_ollama_error", error=str(exc))
        ollama_ok = False

    overall = "ok" if (qdrant_ok and ollama_ok) else "degraded"

    logger.info(
        "health_check",
        status=overall,
        qdrant=qdrant_ok,
        ollama=ollama_ok,
        point_count=point_count,
    )

    return HealthResponse(
        status=overall,
        qdrant="connected" if qdrant_ok else "unreachable",
        ollama="connected" if ollama_ok else "unreachable",
        embedding_model=settings.embedding_model,
        collection=settings.qdrant_collection,
        qdrant_point_count=point_count if qdrant_ok else None,
    )