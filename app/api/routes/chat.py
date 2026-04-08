"""
app/api/routes/chat.py

POST /chat        — main conversation endpoint
POST /chat/report — PDF report upload and analysis endpoint

Both routes are async and properly await the agent.

mode field:
  "concise"  — 3-5 sentence answer, fast (~8-12s locally). Default.
  "detailed" — full explanation, slower (~40-120s locally).
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.models.schemas import ChatResponse
from app.services.agent import get_agent
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    session_id: str = Form(..., description="Unique session identifier"),
    message: str = Form(..., min_length=1, max_length=2000, description="User message"),
    top_k: int = Form(default=5, ge=1, le=20, description="Number of chunks to retrieve"),
    mode: str = Form(default="concise", description="Response mode: 'concise' or 'detailed'"),
):
    logger.info("chat_request", session_id=session_id, message_preview=message[:60], mode=mode)

    if mode not in ("concise", "detailed"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'concise' or 'detailed'",
        )

    try:
        agent = get_agent()
        response = await agent.run(
            session_id=session_id,
            message=message,
            top_k=top_k,
            pdf_bytes=None,
            mode=mode,
        )
        return response

    except Exception as exc:
        logger.error("chat_error", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again.",
        ) from exc


@router.post("/chat/report", response_model=ChatResponse)
async def chat_with_report(
    session_id: str = Form(..., description="Unique session identifier"),
    message: str = Form(
        default="Please explain this medical report in simple terms.",
        max_length=2000,
        description="Optional question about the report",
    ),
    file: UploadFile = File(..., description="PDF medical report to analyse"),
    mode: str = Form(default="concise", description="Response mode: 'concise' or 'detailed'"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    if mode not in ("concise", "detailed"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'concise' or 'detailed'",
        )

    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        logger.error("file_read_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.") from exc

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_size = 10 * 1024 * 1024
    if len(pdf_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum supported size is 10 MB.",
        )

    logger.info(
        "report_upload",
        session_id=session_id,
        filename=file.filename,
        size_kb=round(len(pdf_bytes) / 1024, 1),
        mode=mode,
    )

    try:
        agent = get_agent()
        response = await agent.run(
            session_id=session_id,
            message=message,
            pdf_bytes=pdf_bytes,
            mode=mode,
        )
        return response

    except Exception as exc:
        logger.error("report_error", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail="An error occurred while analysing the report. Please try again.",
        ) from exc