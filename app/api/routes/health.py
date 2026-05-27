"""
app/api/routes/health.py
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
    settings = get_settings()

    # Check Qdrant
    try:
        retriever = get_retrieval_service()
        qdrant_ok, point_count = await retriever.health_check()
    except Exception as exc:
        logger.warning("health_qdrant_error", error=str(exc))
        qdrant_ok, point_count = False, 0

    # Check LLM
    try:
        llm_service = get_llm_service()
        llm_ok = await llm_service.health_check()
    except Exception as exc:
        logger.warning("health_llm_error", error=str(exc))
        llm_ok = False

    overall = "ok" if (qdrant_ok and llm_ok) else "degraded"

    logger.info(
        "health_check",
        status=overall,
        qdrant=qdrant_ok,
        llm=llm_ok,
        point_count=point_count,
    )

    return HealthResponse(
        status=overall,
        qdrant="connected" if qdrant_ok else "unreachable",
        ollama="connected" if llm_ok else "unreachable",
        embedding_model=settings.embedding_model,
        collection=settings.qdrant_collection,
        qdrant_point_count=point_count if qdrant_ok else None,
    )