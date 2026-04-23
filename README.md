# 🏥 MedQ&A RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) system for medical question answering. Built with FastAPI, Qdrant, Mistral AI, and Streamlit — supporting multi-turn Q&A, PDF report analysis, and drug information lookup.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Branches](#branches)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Features](#features)
- [Phase Status](#phase-status)
- [Known Issues](#known-issues)
- [Configuration](#configuration)

---

## Overview

MedQ&A is a local-first RAG chatbot trained on the [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets) (MedQuAD, ~229K Q&A pairs). It supports three inference paths:

| Path | Trigger | Description |
|------|---------|-------------|
| **RAG** | Default | Embeds query → retrieves top-K chunks from Qdrant → Mistral generates grounded answer |
| **Drug Lookup** | Drug name detected | Regex/exclusion-list routing → drugs.com scrape or FDA API fallback → Mistral |
| **Report Analysis** | PDF uploaded | PyMuPDF extraction → embed → Qdrant context → Mistral explanation |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, FastAPI (4 workers) |
| Vector DB | Qdrant (local, port 6333) |
| Embeddings | `all-MiniLM-L6-v2` via Sentence Transformers (384-dim) |
| LLM | Mistral API (`mistral-small-latest`) |
| Orchestration | LangChain |
| Frontend | Streamlit (port 8501) |
| Containerization | Docker + Docker Compose |

---

## Architecture

```
User
 └─► Streamlit (8501)
       └─► api_client.py
             └─► FastAPI /api/v1/chat (8000)
                   └─► agent.py
                         ├─► [RAG]    embed → Qdrant top-8 (reranked) → Mistral
                         ├─► [Drug]   regex detect → drugs.com / FDA API → Mistral
                         └─► [Report] PyMuPDF → embed → Qdrant → Mistral
```

**Retrieval details:**
- Over-fetch + rerank: dense cosine (0.65) + lexical overlap (0.35)
- Candidate pool: 24, final top-k: 8
- Collection: `medquad_chunks` (229,717 points, cosine similarity)

---

## Project Structure

```
phase2_medqa_project/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py          # /chat, /chat/report endpoints
│   │       └── health.py        # /health endpoint
│   ├── services/
│   │   ├── embedding.py         # async sentence transformer wrapper
│   │   ├── retrieval.py         # Qdrant async retrieval + rerank
│   │   ├── llm.py               # async ChatMistralAI wrapper
│   │   └── agent.py             # routing logic + tool orchestration
│   ├── tools/
│   │   ├── drug_tool.py         # drugs.com scraper + FDA fallback
│   │   └── report_tool.py       # PDF extraction + analysis
│   ├── ingestion/
│   │   ├── loader.py            # HuggingFace dataset loader
│   │   ├── cleaner.py           # text normalization
│   │   ├── chunker.py           # token-based chunking (350 tok, 75 overlap)
│   │   └── embedder.py          # batch embedding + Qdrant upsert
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── utils/
│   │   ├── config.py            # settings (Pydantic BaseSettings)
│   │   └── logger.py            # structlog setup
│   ├── frontend/
│   │   ├── app.py               # Streamlit UI (dark clinical theme)
│   │   ├── api_client.py        # httpx sync client
│   │   ├── quality_eval.py      # live heuristic + overlap metrics
│   │   └── test_quality_eval.py # metric unit tests
│   └── dependencies.py          # FastAPI dependency injection
├── main.py                      # FastAPI app entrypoint
├── docker-compose.yml
├── Dockerfile
├── .env                         # secrets (not committed)
├── .env.example                 # template
└── requirements.txt
```

---

## Branches

This repo has **two branches** reflecting different deployment environments:

| Branch | Purpose | LLM | Notes |
|--------|---------|-----|-------|
| `main` | Web / cloud deployment | Mistral API (`mistral-small-latest`) | Uses `web_context.txt` config, full Docker stack |
| `desktop` | Local macOS dev | Ollama (`llama3.1:8b`) | Lighter setup, Streamlit + local Qdrant |

### Switching branches

```bash
# Work on web/cloud branch
git checkout main

# Work on local/desktop branch
git checkout desktop
```

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.com) (desktop branch only)
- Mistral API key (main branch only)
- ~2 GB disk for Qdrant volume (more if re-ingesting)

---

## Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/medqa-rag-chatbot.git
cd medqa-rag-chatbot
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# -- main branch (Mistral) --
MISTRAL_API_KEY=your_mistral_api_key_here
LLM_MODEL=mistral-small-latest

# -- desktop branch (Ollama) --
# OLLAMA_BASE_URL=http://localhost:11434
# LLM_MODEL=llama3.1:8b

# Shared
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=medquad_chunks
API_BASE_URL=http://localhost:8000
```

---

## Running the App

### Option A — Docker (recommended for `main` branch)

```bash
# Start all services
docker-compose up --build -d

# Tail logs
docker-compose logs -f api
docker-compose logs -f frontend

# Stop everything
docker-compose down
```

Services:
- Qdrant: `http://localhost:6333`
- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

---

### Option B — Local (recommended for `desktop` branch)

```bash
# 1. Start Qdrant only via Docker
docker-compose up qdrant -d

# 2. (desktop branch only) Pull and serve the Ollama model
ollama pull llama3.1:8b
ollama serve

# 3. Start FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start Streamlit (new terminal)
streamlit run app/frontend/app.py
```

---

### Data Ingestion (first-time or re-ingestion)

```bash
# Run ingestion pipeline (loads HuggingFace → cleans → chunks → embeds → upserts)
python -m app.ingestion.embedder

# Verify collection
curl http://localhost:6333/collections/medquad_chunks
```

> ⚠️ Re-ingestion is required to benefit from the updated chunking settings (`chunk_size=350`, `overlap=75`). Existing 229,717 points used `chunk_size=400`, `overlap=50`.

---

## API Reference

### `POST /api/v1/chat`

Standard RAG or drug lookup.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -F "session_id=test1" \
  -F "message=What is hypertension?" \
  -F "mode=structured" \
  -F "top_k=8"
```

### `POST /api/v1/chat/report`

PDF medical report analysis.

```bash
curl -X POST http://localhost:8000/api/v1/chat/report \
  -F "session_id=test2" \
  -F "message=Summarize the findings" \
  -F "file=@/path/to/report.pdf"
```

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

**Response modes:** `concise` | `detailed` | `structured`

---

## Features

- **Hybrid retrieval** — dense cosine + lexical overlap reranking
- **Three inference paths** — RAG / drug lookup / PDF report analysis
- **Live quality metrics** — per-response heuristic signals + overlap scores (precision, recall, F1, ROUGE-1/2/L, BLEU) shown in Streamlit UI
- **Retrieval profiles** — High Precision / Balanced / High Recall (adjusts `top_k` and candidate pool)
- **Structured answer mode** — sectioned output with explicit sourcing
- **FDA fallback** — drug tool gracefully falls back to FDA API when drugs.com blocks (403)
- **Configurable LLM params** — temperature (default 0.15), top_p (default 0.9)

---

## Phase Status

| Phase | Description | Status |
|-------|------------|--------|
| 1 | Dataset & Architecture | ✅ Complete |
| 2 | Data Ingestion (229,717 points) | ✅ Complete |
| 3 | Model & API | ✅ Complete |
| 4 | Optimization & Frontend | ✅ Complete |
| 5 | Evaluation | 🔄 In Progress |

**Phase 5 remaining:**
- [ ] Fix health route async bug (`cannot unpack non-iterable coroutine`)
- [ ] Re-ingest with new chunk settings and run A/B evaluation
- [ ] Compute formal metrics (Precision@K, Recall@K, MRR, NDCG, ROUGE, BLEU) against MedQuAD ground truth
- [ ] Finalize evaluation report

---

## Known Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| Health route async bug | Health panel may show incorrect status | Await async calls in `health.py` |
| `drugs.com` 403 | Drug tool falls back to FDA API (works, thinner response) | Planned: ingest drug data into Qdrant |
| Overlap metrics are low | Context-grounding only, not gold-answer correctness | Re-ingest with smaller chunks |
| Docker: only Qdrant auto-starts | API/frontend need manual start in Docker | Healthcheck config TBD |

---

## Configuration

All settings live in `app/utils/config.py` (Pydantic `BaseSettings`, reads from `.env`):

| Key | Default | Description |
|-----|---------|-------------|
| `llm_model` | `mistral-small-latest` | LLM model name |
| `llm_temperature` | `0.15` | Lower = less hallucination |
| `llm_top_p` | `0.9` | Nucleus sampling |
| `retrieval_top_k` | `8` | Final retrieved chunks |
| `retrieval_candidate_pool` | `24` | Over-fetch before rerank |
| `retrieval_keyword_weight` | `0.35` | Lexical score weight |
| `retrieval_dense_weight` | `0.65` | Dense cosine weight |
| `chunk_size` | `350` | Tokens per chunk (next ingestion) |
| `chunk_overlap` | `75` | Token overlap (next ingestion) |

---

## License

MIT
