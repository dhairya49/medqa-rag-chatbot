"""
app/main.py

FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app instance
  - Register startup / shutdown lifespan events
    (pre-warm all singletons so first request is not slow)
  - Add CORS middleware
  - Register all routers under /api/v1
  - Expose the app object for uvicorn

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run via Docker:
    docker-compose up --build

Depends on:
  - app/api/routes/chat.py    (chat + report upload endpoints)
  - app/api/routes/health.py  (health check endpoint)
  - app/utils/config.py       (get_settings)
  - app/utils/logger.py       (setup_logging, get_logger)
  - all service singletons    (pre-warmed in lifespan)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.utils.config import get_settings
from app.utils.logger import setup_logging, get_logger


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown context manager.

    On startup:
      - Initialise structured logging
      - Pre-warm all singleton services so the first real request
        doesn't pay the cold-start cost (model load, Qdrant connect, Ollama connect)

    On shutdown:
      - Log graceful shutdown (connections are cleaned up by GC)
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging()
    logger = get_logger("startup")
    settings = get_settings()

    logger.info(
        "app_starting",
        host=settings.app_host,
        port=settings.app_port,
        llm_model=settings.llm_model,
        ollama_host=settings.ollama_host,
        qdrant_host=settings.qdrant_host,
        collection=settings.qdrant_collection,
    )

    # Pre-warm embedding model (takes 2-3s — SentenceTransformer load)
    logger.info("prewarm_embedding_model")
    from app.services.embedding import get_embedding_service
    get_embedding_service()

    # Pre-warm Qdrant connection
    logger.info("prewarm_qdrant")
    from app.services.retrieval import get_retrieval_service
    get_retrieval_service()

    # Pre-warm Ollama connection
    logger.info("prewarm_ollama")
    from app.services.llm import get_llm_service
    get_llm_service()

    # Pre-warm agent (pulls in all the above singletons)
    logger.info("prewarm_agent")
    from app.services.agent import get_agent
    get_agent()

    logger.info("app_ready")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("app_shutting_down")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Separated from module level so it can be imported cleanly in tests.
    """
    settings = get_settings()

    app = FastAPI(
        title="MedQA RAG Chatbot",
        description=(
            "A medical question-answering chatbot powered by RAG. "
            "Retrieves verified medical information from MedQuAD, "
            "analyses uploaded medical reports, and provides live drug information."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI at /docs
        redoc_url="/redoc",     # ReDoc at /redoc
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow all origins for development.
    # In Phase 4 / production, restrict to your actual frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(health_router, prefix="/api/v1", tags=["health"])

    return app


# ── App instance ──────────────────────────────────────────────────────────────
# This is what uvicorn imports: uvicorn app.main:app

app = create_app()