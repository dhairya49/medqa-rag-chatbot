# ── Base image ────────────────────────────────────────────────────────────────
# Using 3.11-slim — stable, well tested with all dependencies
# (local dev uses 3.14 but 3.11 is fine for production container)
FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model ───────────────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── App source ────────────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entry point ───────────────────────────────────────────────────────────────
# app.main:app — matches your project structure (app/main.py, app = create_app())
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]